import os
from os.path import join as opj
import json
import pickle
import math
from glob import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TF
from safetensors.torch import load_file as sf_load_file

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def zero_rank_print_(s):
    if "LOCAL_RANK" in os.environ.keys():
        if int(os.environ["LOCAL_RANK"]) == 0: 
            print(s)
    else:
        print(s)

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
        
def load_args(from_path):
    with open(from_path, "r") as f:
        args_dict = json.load(f)
    return args_dict

def load_file(p):
    if p.endswith(".safetensors"):
        cp = sf_load_file(p)
    else:
        cp = torch.load(p, map_location="cpu")
    return cp

def tensor2pil(tensor, is_mask=False):
    tensor = tensor.cpu()
    if is_mask:
        return Image.fromarray(np.uint8(tensor[0][0].numpy() * 255)).convert("RGB")
    else:
        tensor = (tensor[0].permute(1,2,0)+1) * 127.5
        return Image.fromarray(np.uint8(tensor))

def concat_pil_imgs(pil_img_lst):
    max_img_h = -1
    ratio_lst = []
    for pil_img in pil_img_lst:
        img_w, img_h = pil_img.size
        max_img_h = max(max_img_h, img_h)
        ratio_lst.append(img_w / img_h)

    
    new_img_lst = []
    for pil_img, ratio in zip(pil_img_lst, ratio_lst):
        np_img = np.array(pil_img.resize((int(ratio * max_img_h), max_img_h)))
        if np_img.ndim == 2:
            np_img = np.stack([np_img] * 3, axis=-1)
        if np_img.shape[-1] == 1:
            np_img = np.concatenate([np_img]*3, axis=-1)
        new_img_lst.append(np_img)
    
    concat_img = np.concatenate(new_img_lst, axis=1)
    return Image.fromarray(concat_img)

@torch.no_grad()
def get_attn_map(hidden_states, encoder_hidden_states, attn, norm_axis=-1):
    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_wieght_logit = attn_weight
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.cpu().mean(dim=1)

    
    min_ = attn_weight.min(dim=norm_axis, keepdims=True)[0]
    max_ = attn_weight.max(dim=norm_axis, keepdims=True)[0]
    norm_attn_weight = (attn_weight - min_) / (max_ - min_) * 255.0
    norm_attn_weight = norm_attn_weight.numpy().astype(np.uint8)
    return norm_attn_weight, attn_wieght_logit

def pad_resize(img, trg_h, trg_w, pixel_value, pad_type=None):
    if pad_type is None:
        img = img.resize((trg_w, trg_h))
    else:
        cur_w, cur_h = img.size
        pad_w = max(trg_w - cur_w, 0)
        pad_h = max(trg_h - cur_h, 0)

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        img = TF.pad(img, padding=padding, fill=pixel_value, padding_mode=pad_type)
    return img
        
def get_inputs(
        root_dir, data_type, pose_type, img_bn, c_bn, img_h, img_w, train_folder_name, test_folder_name, 
        # use_repaint, train_folder_name_for_interm_cloth_mask=None, test_repaint_folder_name=None,
        # return_inversion_latents=False,
        category=None, pad_type=None, use_dc_cloth=False
):
    is_vitonhd = category is None or category == ""
    img_fn = os.path.splitext(img_bn)[0]
    if is_vitonhd:
        if data_type == "train": 
            folder_name = train_folder_name if train_folder_name is not None else "train"
        else: 
            folder_name = test_folder_name if test_folder_name is not None else "test"
        
        person = Image.open(opj(root_dir, f"{folder_name}/image", img_bn)).convert("RGB").resize((img_w, img_h))
        mask = Image.open(opj(root_dir, f"{folder_name}/agnostic-mask", f"{img_fn}_mask.png")).convert("RGB").resize((img_w, img_h))
        cloth = Image.open(opj(root_dir, f"{folder_name}/cloth", c_bn)).convert("RGB").resize((img_w, img_h))

        if pose_type == "openpose": pose = Image.open(opj(root_dir, f"{folder_name}/dwpose", f"{img_fn}.png")).convert("RGB").resize((img_w, img_h))
        elif pose_type == "openpose_thick": pose = Image.open(opj(root_dir, f"{folder_name}/dwpose_thick", f"{img_fn}.png")).convert("RGB").resize((img_w, img_h))
        elif pose_type == "densepose": pose = Image.open(opj(root_dir, f"{folder_name}/image-densepose", f"{img_fn}.jpg")).convert("RGB").resize((img_w, img_h))
            
        person = Image.open(opj(root_dir, f"{folder_name}/image", img_bn)).convert("RGB")
        mask = Image.open(opj(root_dir, f"{folder_name}/agnostic-mask", f"{img_fn}_mask.png")).convert("RGB")
        if not use_dc_cloth:
            cloth = Image.open(opj(root_dir, f"{folder_name}/cloth", c_bn)).convert("RGB")
        else:
            cloth = Image.open(opj(root_dir, f"{folder_name}/cloth_dc", c_bn)).convert("RGB")

        if pose_type == "openpose": pose = Image.open(opj(root_dir, f"{folder_name}/dwpose", f"{img_fn}.png")).convert("RGB")
        elif pose_type == "openpose_thick": pose = Image.open(opj(root_dir, f"{folder_name}/dwpose_thick", f"{img_fn}.png")).convert("RGB")
        elif pose_type == "densepose": pose = Image.open(opj(root_dir, f"{folder_name}/image-densepose", f"{img_fn}.jpg")).convert("RGB")

        person = pad_resize(person, img_h, img_w, (255,255,255), pad_type=pad_type)
        if pad_type is None or pad_type == "resize":
            other_pad_type = None
        else:
            other_pad_type = "constant"
        mask = pad_resize(mask, img_h, img_w, (0,0,0), pad_type=other_pad_type)
        cloth = pad_resize(cloth, img_h, img_w, (255,255,255), pad_type=other_pad_type)
        pose = pad_resize(pose, img_h, img_w, (0,0,0), pad_type=other_pad_type)

    return person, mask, pose, cloth

def get_leanable_param_count(model_name, model):
    named_param = model.named_parameters()
    total_count = 0
    lparam_count = 0
    not_lparam_count = 0
    for name, param in named_param:
        if param.requires_grad:
            lparam_count += 1
        else:
            not_lparam_count += 1
        total_count += 1
    return f"  {model_name} | total : {total_count}, lparam : {lparam_count}, not_lparam : {not_lparam_count}"

def split_procidx(ps, n_proc, proc_idx):
    len_ps = len(ps)
    if len_ps % n_proc == 0:
        n_infer = len_ps // n_proc
    else:
        n_infer = len_ps // n_proc + 1
    
    start_idx = int(proc_idx * n_infer)
    end_idx = start_idx + n_infer
    ps = ps[start_idx:end_idx]
    return ps

def get_tensor(img, h, w, is_mask=False):
    img = np.array(img.resize((w, h))).astype(np.float32)
    if not is_mask:
        img = (img / 127.5) - 1.0
    else:
        img = (img < 128).astype(np.float32)[:,:,None]
    return torch.from_numpy(img)[None].cuda()
    
def get_batch(image, cloth, densepose, agn_img, agn_mask, img_h, img_w):
    batch = dict()
    batch["image"] = get_tensor(image, img_h, img_w)
    batch["cloth"] = get_tensor(cloth, img_h, img_w)
    batch["image_densepose"] = get_tensor(densepose, img_h, img_w)
    batch["agn"] = get_tensor(agn_img, img_h, img_w)
    batch["agn_mask"] = get_tensor(agn_mask, img_h, img_w, is_mask=True)
    batch["txt"] = ""
    return batch

def tensor2img(x):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    x = np.uint8(x*255.0)
    if x.shape[-1] == 1:
        x = np.concatenate([x,x,x], axis=-1)
    return x

def center_crop(image):
    width, height = image.size
    new_height = height
    new_width = height*3/4
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))
    return image

def get_lora_target_modules(named_modules, all_names, any_names, not_names):
    output = []
    lora_modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d]
    for key, module in named_modules:
        if all(all_name in key for all_name in all_names) and any(any_name in key for any_name in any_names) and not any(not_name in key for not_name in not_names):
            for lora_module in lora_modules:
                if isinstance(module, lora_module):
                    output.append(key)
    return output

    
def unfreeze_unet(unet, all_names, any_names, not_names):
    for key, param in unet.named_parameters():
        if all(all_name in key for all_name in all_names) and any(any_name in key for any_name in any_names) and not any(not_name in key for not_name in not_names):
            param.requires_grad_(True)


def get_txt(jf, person_id, clothing_id=None, prompt_version="v5", category="upper_body", verbose=True):
    from .data.data_utils import Prompter
    pt = Prompter(category=category, version=prompt_version)
    if clothing_id is None:
        clothing_id = person_id
    person_dict = jf[person_id]["person"]
    clothing_dict = jf[clothing_id]["clothing"]
    clothing_person_dict = jf[clothing_id]["person"]
    full_txt, clothing_txt = pt.generate(person_dict, clothing_dict, clothing_person_dict)
    if verbose:
        print(full_txt)
        print("\n")
        print(clothing_txt)
        print("\n\n")

def concat_save_images(ps_lst, save_dir, cut_right_two=False):
    import cv2
    from tqdm import tqdm
    os.makedirs(save_dir, exist_ok=True)

    min_value = min([len(ps) for ps in ps_lst])
    for i in tqdm(range(min_value), total=min_value):
        concat = []
        for ps in ps_lst:
            p = ps[i]
            concat.append(cv2.imread(p))
        concat = np.concatenate(concat, axis=1)
        if cut_right_two:
            concat = concat[:,:-2*768]
        
        save_p = opj(save_dir, os.path.basename(p))
        cv2.imwrite(save_p, concat)