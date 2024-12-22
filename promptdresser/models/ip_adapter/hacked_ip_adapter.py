import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder, image_proj_model):
        self.pipe = sd_pipe
        self.device = sd_pipe.device
        self.image_encoder = image_encoder
        self.image_proj_model = image_proj_model
        self.clip_image_processor = CLIPImageProcessor()

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
    
class IPAdapterInpainting(IPAdapter):
    def generate(
        self,
        unet_image,  # person
        mask_image,  # agn mask
        pose_image,  # pose
        ip_adapter_image,  # cloth
        cloth_encoder,
        cloth_encoder_image,  # cloth
        prompt=None,
        prompt_clothing=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        num_inference_steps=30,
        strength=1.0,
        guidance_scale=2.0,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(ip_adapter_image, Image.Image) else len(ip_adapter_image)

        if prompt is None:
            prompt = "best quality, high quality"
            prompt_clothing = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
            prompt_clothing = [prompt_clothing] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(ip_adapter_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds
            ) = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

            (
                prompt_embeds_clothing,
                _
            ) = self.pipe.encode_prompt(
                prompt_clothing,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_clothing = torch.cat([prompt_embeds_clothing, image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            image=unet_image,
            mask_image=mask_image,
            pose_image=pose_image,
            prompt_embeds=prompt_embeds,
            prompt_embeds_clothing=prompt_embeds_clothing,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            cloth_encoder=cloth_encoder,
            cloth_encoder_image=cloth_encoder_image,
            strength=strength,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        return images
    

class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images
    
class IPAdapterXLInpainting(IPAdapter):
    """SDXL"""

    def generate(
        self,
        unet_image,  # person
        mask_image,  # agn mask
        pose_image,  # pose
        ip_adapter_image,  # cloth
        cloth_encoder,
        cloth_encoder_image,  # cloth
        prompt=None,
        prompt_clothing=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        num_inference_steps=30,
        strength=1.0,
        guidance_scale=2.0,
        **kwargs,
    ):
        num_prompts = 1 if isinstance(ip_adapter_image, Image.Image) else len(ip_adapter_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if prompt_clothing is None:
            prompt_clothing = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
            prompt_clothing = [prompt_clothing] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                save_eos=True,
            )

            (
                prompt_embeds_clothing,
                _,
                pooled_prompt_embeds_clothing,
                _,
            ) = self.pipe.encode_prompt(
                prompt_clothing,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        if self.image_proj_model is not None:
            self.set_scale(scale)
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(ip_adapter_image)
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            prompt_embeds_clothing = torch.cat([prompt_embeds_clothing, image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

    
        images = self.pipe(
            image=unet_image,
            mask_image=mask_image,
            pose_image=pose_image,
            prompt_embeds=prompt_embeds,
            prompt_embeds_clothing=prompt_embeds_clothing,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            pooled_prompt_embeds_clothing=pooled_prompt_embeds_clothing,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            cloth_encoder=cloth_encoder,
            cloth_encoder_image=cloth_encoder_image,
            strength=strength,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        return images
    





    def get_interm_clothmask(
        self,
        unet_image,  # person
        mask_image,  # agn mask
        pose_image,  # pose
        ip_adapter_image,  # cloth
        cloth_encoder,
        cloth_encoder_image,  # cloth
        prompt=None,
        prompt_clothing=None,
        negative_prompt="",
        scale=1.0,
        num_samples=1,
        seed=None,
        num_inference_steps=30,
        strength=1.0,
        guidance_scale=2.0,
        **kwargs,
    ):
        num_prompts = 1 if isinstance(ip_adapter_image, Image.Image) else len(ip_adapter_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if prompt_clothing is None:
            prompt_clothing = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
            prompt_clothing = [prompt_clothing] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                save_eos=True,
            )

            (
                prompt_embeds_clothing,
                _,
                pooled_prompt_embeds_clothing,
                _,
            ) = self.pipe.encode_prompt(
                prompt_clothing,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        if self.image_proj_model is not None:
            self.set_scale(scale)
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(ip_adapter_image)
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            prompt_embeds_clothing = torch.cat([prompt_embeds_clothing, image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

    
        images = self.pipe.get_interm_clothmask(
            image=unet_image,
            mask_image=mask_image,
            pose_image=pose_image,
            prompt_embeds=prompt_embeds,
            prompt_embeds_clothing=prompt_embeds_clothing,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            pooled_prompt_embeds_clothing=pooled_prompt_embeds_clothing,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            cloth_encoder=cloth_encoder,
            cloth_encoder_image=cloth_encoder_image,
            strength=strength,
            guidance_scale=guidance_scale,
            **kwargs,
        )

        return images
    


    
# class IPAdapterXLInpainting(IPAdapter):
#     """SDXL"""

#     def generate(
#         self,
#         unet_image,  # person
#         mask_image,  # agn mask
#         pose_image,  # pose
#         ip_adapter_image,  # cloth
#         cloth_encoder,
#         cloth_encoder_image,  # cloth
#         prompt=None,
#         prompt_clothing=None,
#         negative_prompt=None,
#         scale=1.0,
#         num_samples=1,
#         seed=None,
#         num_inference_steps=30,
#         strength=1.0,
#         guidance_scale=2.0,
#         **kwargs,
#     ):
#         self.set_scale(scale)

#         num_prompts = 1 if isinstance(ip_adapter_image, Image.Image) else len(ip_adapter_image)

#         if prompt is None:
#             prompt = "best quality, high quality"
#         if prompt_clothing is None:
#             prompt_clothing = "best quality, high quality"
#         if negative_prompt is None:
#             negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

#         if not isinstance(prompt, List):
#             prompt = [prompt] * num_prompts
#             prompt_clothing = [prompt_clothing] * num_prompts
#         if not isinstance(negative_prompt, List):
#             negative_prompt = [negative_prompt] * num_prompts

#         image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(ip_adapter_image)
#         bs_embed, seq_len, _ = image_prompt_embeds.shape
#         image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
#         image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

#         with torch.inference_mode():
#             (
#                 prompt_embeds,
#                 negative_prompt_embeds,
#                 pooled_prompt_embeds,
#                 negative_pooled_prompt_embeds,
#             ) = self.pipe.encode_prompt(
#                 prompt,
#                 num_images_per_prompt=num_samples,
#                 do_classifier_free_guidance=True,
#                 negative_prompt=negative_prompt,
#             )
#             prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
#             negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

#             (
#                 prompt_embeds_clothing,
#                 _,
#                 pooled_prompt_embeds_clothing,
#                 _,
#             ) = self.pipe.encode_prompt(
#                 prompt_clothing,
#                 num_images_per_prompt=num_samples,
#                 do_classifier_free_guidance=True,
#                 negative_prompt=negative_prompt,
#             )
#             prompt_embeds_clothing = torch.cat([prompt_embeds_clothing, image_prompt_embeds], dim=1)

#         self.generator = get_generator(seed, self.device)
        
#         images = self.pipe(
#             image=unet_image,
#             mask_image=mask_image,
#             pose_image=pose_image,
#             prompt_embeds=prompt_embeds,
#             prompt_embeds_clothing=prompt_embeds_clothing,
#             negative_prompt_embeds=negative_prompt_embeds,
#             pooled_prompt_embeds=pooled_prompt_embeds,
#             pooled_prompt_embeds_clothing=pooled_prompt_embeds_clothing,
#             negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#             num_inference_steps=num_inference_steps,
#             generator=self.generator,
#             cloth_encoder=cloth_encoder,
#             cloth_encoder_image=cloth_encoder_image,
#             strength=strength,
#             guidance_scale=guidance_scale,
#             **kwargs,
#         ).images

#         return images

class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
