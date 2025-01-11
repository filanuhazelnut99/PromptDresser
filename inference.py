import os
from os.path import join as opj
import argparse
import gc

from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPTextModelWithProjection
)
from accelerate import Accelerator

from promptdresser.utils import (
    zero_rank_print_, 
    get_inputs,
    load_file
)
from promptdresser.data.data_utils import get_validation_pairs
from promptdresser.models.unet import UNet2DConditionModel
from promptdresser.models.cloth_encoder import ClothEncoder
from promptdresser.models.mutual_self_attention import ReferenceAttentionControl
from promptdresser.pipelines.sdxl import PromptDresser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_p", type=str, required=True)
    parser.add_argument("--noise_scheduler_name", default="ddpm")
    parser.add_argument("--timestep_spacing", type=str, default="leading", choices=["leading", "trailing"])
    parser.add_argument("--interm_cloth_start_ratio", type=float, default=None)
    parser.add_argument("--pretrained_unet_path", type=str, default=None)
    parser.add_argument("--pretrained_cloth_encoder_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--skip_paired", action="store_true")
    parser.add_argument("--skip_unpaired", action="store_true")
    parser.add_argument("--n_repeat_samples", type=int, default=1)
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--e_idx", type=int, default=999999)
    parser.add_argument("--pad_type", default=None, choices=["constant", "reflect"])
    parser.add_argument("--save_root_dir", type=str, default="./sampled_images")
    parser.add_argument("--save_name", type=str, default="dummy")

    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--no_zero_snr", action="store_true")
    parser.add_argument("--img_h", type=int, default=1024)
    parser.add_argument("--img_w", type=int, default=768)
    parser.add_argument("--init_model_path", type=str, default="./pretrained_models/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--init_vae_path", type=str, default="./pretrained_models/sdxl-vae-fp16-fix")
    parser.add_argument("--init_cloth_encoder_path", type=str, default="./pretrained_models/stable-diffusion-xl-base-1.0")
    parser.add_argument("--ip_adapter_num_tokens", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--guidance_scale_img", type=float, default=4.5)
    parser.add_argument("--guidance_scale_text", type=float, default=7.5)
    parser.add_argument("--mixed_precision", type=str, default="fp16")

    args = parser.parse_args()
    args.save_dir = opj(args.save_root_dir, args.save_name)
    os.makedirs(args.save_dir, exist_ok=True)
    return args

args = parse_args()
config = OmegaConf.load(args.config_p)
if args.interm_cloth_start_ratio is not None:
    config.interm_cloth_start_ratio = args.interm_cloth_start_ratio
accelerator = Accelerator(mixed_precision=args.mixed_precision)
weight_dtype = torch.float16


noise_scheduler = DDPMScheduler.from_pretrained(
    args.init_model_path, subfolder="scheduler", 
    rescale_betas_zero_snr=not args.no_zero_snr, 
    timestep_spacing=args.timestep_spacing
)
tokenizer = CLIPTokenizer.from_pretrained(args.init_model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(args.init_model_path, subfolder="text_encoder")
tokenizer_2 = CLIPTokenizer.from_pretrained(args.init_model_path, subfolder="tokenizer_2")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.init_model_path, subfolder="text_encoder_2")
vae = AutoencoderKL.from_pretrained(args.init_vae_path)
unet = UNet2DConditionModel.from_pretrained(args.init_model_path, subfolder="unet")
cloth_encoder = ClothEncoder.from_pretrained(args.init_cloth_encoder_path, subfolder="unet")

unet.add_clothing_text = False

unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
cloth_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)

unet.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)
text_encoder_2.to(accelerator.device, dtype=weight_dtype)
cloth_encoder.to(accelerator.device, dtype=weight_dtype)

if not config.get("detach_cloth_encoder", False):
    reference_control_writer = ReferenceAttentionControl(
        cloth_encoder, 
        do_classifier_free_guidance=True, 
        mode="write", fusion_blocks="midup" if os.environ.get("MIDUP_FUSION_BLOCK", False) else "full",
        batch_size=1, 
        is_train=True,
        is_second_stage=False, 
        use_jointcond=config.get("use_jointcond", False)
    )
    reference_control_reader = ReferenceAttentionControl(
        unet, 
        do_classifier_free_guidance=True, 
        mode="read", 
        fusion_blocks="midup" if os.environ.get("MIDUP_FUSION_BLOCK", False) else "full", 
        batch_size=1, 
        is_train=True, 
        is_second_stage=False, 
        use_jointcond=config.get("use_jointcond", False)
    )

if args.pretrained_unet_path is not None:
    unet.load_state_dict(load_file(args.pretrained_unet_path))
    zero_rank_print_(f"unet is loaded from {args.pretrained_unet_path}")

if args.pretrained_cloth_encoder_path is not None:
    cloth_encoder.load_state_dict(load_file(args.pretrained_cloth_encoder_path), strict=False)
    zero_rank_print_(f"cloth_encoder is loaded from {args.pretrained_cloth_encoder_path}")
    
pipeline = PromptDresser(
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=noise_scheduler,
).to(accelerator.device, dtype=weight_dtype)

pipeline.set_progress_bar_config(leave=False)



dataset_config = config.dataset
pair_type_lst = []
img_bns_lst = []
c_bns_lst = []
full_txts_lst = []
clothing_txts_lst = []

if not args.skip_paired:
    paired_img_bns, paired_c_bns, paired_full_txts, paired_clothing_txts = get_validation_pairs(
        **dataset_config, is_paired=True, proc_idx=accelerator.process_index, n_proc=accelerator.num_processes, data_type="test", 
    )
    pair_type_lst.append("paired")
    img_bns_lst.append(paired_img_bns)
    c_bns_lst.append(paired_c_bns)
    full_txts_lst.append(paired_full_txts)
    clothing_txts_lst.append(paired_clothing_txts)

if not args.skip_unpaired:
    unpaired_img_bns, unpaired_c_bns, unpaired_full_txts, unpaired_clothing_txts = get_validation_pairs(
        **dataset_config, is_paired=False, proc_idx=accelerator.process_index, n_proc=accelerator.num_processes, data_type="test", 
    )
    pair_type_lst.append("unpaired")
    img_bns_lst.append(unpaired_img_bns)
    c_bns_lst.append(unpaired_c_bns)
    full_txts_lst.append(unpaired_full_txts)
    clothing_txts_lst.append(unpaired_clothing_txts)

for idx, (pair_type, img_bns, c_bns, full_txts, clothing_txts) in enumerate(zip(pair_type_lst, img_bns_lst, c_bns_lst, full_txts_lst, clothing_txts_lst)):  
    if args.skip_paired and pair_type=="paired":
        zero_rank_print_("skip paired")
        continue
    if args.skip_unpaired and pair_type=="unpaired": 
        zero_rank_print_("skip unpaired")
        continue

    img_save_dir = opj(args.save_dir, pair_type)
    os.makedirs(img_save_dir, exist_ok=True)

    with tqdm(enumerate(zip(img_bns, c_bns, full_txts, clothing_txts)), total=len(img_bns), unit="iter", ncols=75) as tl:
        for sample_idx, (img_bn, c_bn, full_txt, clothing_txt) in tl:
            if not (args.s_idx <= sample_idx < args.e_idx):
                print(f"skip {sample_idx}")
                continue

            tl.set_description(f"sample {pair_type}")
            for repeat_idx in range(args.n_repeat_samples):
                img_fn = os.path.splitext(img_bn)[0]
                c_fn = os.path.splitext(c_bn)[0]
                if args.n_repeat_samples == 1:
                    to_p = opj(img_save_dir, f"{img_fn}__{c_fn}.jpg")
                else:
                    to_p = opj(img_save_dir, f"{img_fn}__{c_fn}_{repeat_idx}.jpg")                

                person, mask, pose, cloth = get_inputs(
                    root_dir=dataset_config.data_root_dir,
                    data_type="test",
                    pose_type=dataset_config.pose_type,
                    img_bn=img_bn,
                    c_bn=c_bn,
                    img_h=args.img_h,
                    img_w=args.img_w,
                    train_folder_name=dataset_config.get("train_folder_name", None),
                    test_folder_name=dataset_config.get("test_folder_name", None),
                    category=dataset_config.get("category", None),
                    pad_type=args.pad_type,
                    use_dc_cloth=dataset_config.get("use_dc_cloth", False),
                )
                
                if config.get("use_interm_cloth_mask", False):
                    _person, _mask, _pose, _cloth = get_inputs(
                        root_dir=dataset_config.data_root_dir,
                        data_type="test",
                        pose_type=dataset_config.pose_type,
                        img_bn=img_bn,
                        c_bn=c_bn,
                        img_h=args.img_h,
                        img_w=args.img_w,
                        train_folder_name=dataset_config.train_folder_name_for_interm_cloth_mask,
                        test_folder_name=dataset_config.test_folder_name_for_interm_cloth_mask,
                        category=dataset_config.get("category", None),
                        pad_type=args.pad_type,
                        use_dc_cloth=dataset_config.get("use_dc_cloth", False),
                    )
                    with torch.autocast("cuda"):
                        interm_cloth_mask = pipeline.get_interm_clothmask(
                                image=_person, 
                                mask_image=_mask,
                                pose_image=_pose,
                                cloth_encoder=cloth_encoder,
                                cloth_encoder_image=cloth,
                                prompt=full_txt,
                                prompt_clothing=clothing_txt,
                                height=args.img_h, 
                                width=args.img_w,
                                guidance_scale=args.guidance_scale,
                                guidance_scale_img=args.guidance_scale_img,
                                guidance_scale_text=args.guidance_scale_text,
                                num_inference_steps=args.num_inference_steps,
                                use_jointcond=config.get("use_jointcond", False),
                                interm_cloth_start_ratio=config.get("interm_cloth_start_ratio", 0.5),
                                detach_cloth_encoder=config.get("detach_cloth_encoder", False),
                                strength=args.strength,
                                category=dataset_config.get("category", None),
                                use_pad=config.get("interm_cloth_pad", False),
                                generator = None,
                        )

                        interm_cloth_mask = interm_cloth_mask.resize((args.img_w, args.img_h), Image.NEAREST)
                        mask = Image.fromarray(np.maximum(np.array(mask), np.array(interm_cloth_mask)[:,:,None]))
                
                with torch.autocast("cuda"):
                    print(f"full txt : {full_txt}")
                    print(f"clothing txt : {clothing_txt}")
                    sample = pipeline(
                        image=person, 
                        mask_image=mask,
                        pose_image=pose,
                        cloth_encoder=cloth_encoder,
                        cloth_encoder_image=cloth,
                        prompt=full_txt,
                        prompt_clothing=clothing_txt,
                        height=args.img_h, 
                        width=args.img_w,
                        guidance_scale=args.guidance_scale,
                        guidance_scale_img=args.guidance_scale_img,
                        guidance_scale_text=args.guidance_scale_text,
                        num_inference_steps=args.num_inference_steps,
                        use_jointcond=config.get("use_jointcond", False),
                        interm_cloth_start_ratio=config.get("interm_cloth_start_ratio", 0.5),
                        detach_cloth_encoder=config.get("detach_cloth_encoder", False),
                        strength=args.strength,
                        category=dataset_config.get("category", None),
                        generator = None,
                    ).images[0]
                    
                    if args.pad_type:
                        sample = sample.crop((128, 0, 640, 1024))

                    inverse_mask = np.array(mask) < 0.5
                    agn_img = Image.fromarray(np.uint8(np.array(person) * inverse_mask.astype(np.float32)))
                    sample.save(to_p)


accelerator.wait_for_everyone()
torch.cuda.empty_cache()
gc.collect()
if args.skip_paired:
    pair_type_lst = ["unpaired"]
elif args.skip_unpaired:
    pair_type_lst = ["paired"]
else:
    pair_type_lst = ["paired", "unpaired"]

if accelerator.is_main_process:
    for pair_type in pair_type_lst:
        img_save_dir = opj(args.save_dir, pair_type)
        gt_dir = "./DATA/zalando-hd-resized/test_fine/image"
    
        eval_cmd = f"python evaluation.py --gt_dir {gt_dir} --pred_dir {img_save_dir} &"
        os.system(eval_cmd)
    print("Done")