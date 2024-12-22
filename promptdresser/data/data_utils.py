import os, json
from os.path import join as opj

import numpy as np
from scipy.ndimage import binary_dilation
import cv2
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

from ..utils import split_procidx

def remove_postfix(fn):
    splits = fn.split("_")
    if len(splits) > 3: 
        return fn
    else:
        return splits[0]
    
def get_validation_pairs(
        data_root_dir, 
        is_paired, 
        data_type,
        category=None,
        proc_idx=0, 
        n_proc=1, 
        n_samples=None,
        prompt_version=None,
        text_file_postfix=None,
        use_dc_cloth=False,
        test_file_postfix=None,
        **kwargs,
    ):
    img_names = []
    c_names = []
    full_texts = []
    clothing_texts = []
    is_vitonhd = category is None
    assert data_type in ["train", "test"]
    if data_type == "train":
        pair_postfix = "pairs"
    else:
        if is_paired:
            pair_postfix = "pairs"
        else:
            pair_postfix = "unpairs"
    if test_file_postfix is not None:
        print(f"pair postfix : {test_file_postfix}")
        pair_postfix = test_file_postfix
    if not is_vitonhd:
        assert category in ["upper_body", "lower_body", "dresses"]
        data_root_dir = opj(data_root_dir, category)

    if not use_dc_cloth:
        txt_path = opj(data_root_dir, f"{data_type}_{pair_postfix}.txt")
    else:
        txt_path = opj(data_root_dir, f"{data_type}_{pair_postfix}_dc.txt")
        
    with open(txt_path, "r") as f:
        for line in f.readlines():
            img_name, c_name = line.strip().split()
            img_names.append(img_name)
            c_names.append(c_name)
    img_names, c_names = map(list, zip(*sorted(zip(img_names, c_names))))
    img_names = split_procidx(img_names, n_proc, proc_idx)
    c_names = split_procidx(c_names, n_proc, proc_idx)
    img_names = img_names[:n_samples]
    c_names = c_names[:n_samples]

    prompter = Prompter(category="upper_body" if is_vitonhd else category, version=prompt_version, data_type=data_type)
    if text_file_postfix is not None:
        if is_vitonhd:  # vitonhd
            textfile_bn = f"{data_type}_{text_file_postfix}"
        else:
            textfile_bn = text_file_postfix

        with open(opj(data_root_dir, textfile_bn), "rb") as f:
            text_dict = json.load(f)
    
    if text_file_postfix is None:
        full_texts = ["" for _ in range(len(img_names))]
        clothing_texts = ["" for _ in range(len(img_names))]
    else:
        for img_name, c_name in zip(img_names, c_names):
            if (category == "upper_body") and ("012143_0" in img_name):  # error
                continue
            
            img_fn = os.path.splitext(img_name)[0]
            c_fn = os.path.splitext(c_name)[0]
            if not is_vitonhd:
                img_fn = remove_postfix(img_fn)
                c_fn = remove_postfix(c_fn)
            
            person_dict = text_dict[img_fn]["person"]
            clothing_dict = text_dict[c_fn]["clothing"]
            if "person" in text_dict[c_fn].keys():
                clothing_person_dict = text_dict[c_fn]["person"]
            else:  
                clothing_person_dict = person_dict
            full_txt, clothing_txt = prompter.generate(person_dict, clothing_dict, clothing_person_dict)
                
            full_texts.append(full_txt)
            clothing_texts.append(clothing_txt)
    return img_names, c_names, full_texts, clothing_texts


class Prompter:
    def __init__(self, category, version, data_type="train"):
        assert category in ["upper_body", "lower_body", "dresses"]
        self.category = category
        self.version = version
        self.data_type = data_type 
        print(f"category : {self.category}, version : {self.version}")

    @staticmethod
    def create_prompt_v12(person_dict, clothing_dict, clothing_person_dict): 
        clothing_template = "a {category}, {material}, with {sleeve}, {neckline}"
        full_template = "a {body_shape} {gender} wears {fit_of_clothing}, {category} ({material}), {neckline}, {sleeve_rolling_style}, {tucking_style}. With {hair_length} hair, {pose} with hands {hand_pose}"
        
        # tucking style
        if "crop" in clothing_dict["upper cloth length"]:
            tucking_style = "untucked"
        else:
            tucking_style = person_dict["tucking style"]

        if "short" in clothing_dict["sleeve"]:
            sleeve_rolling_style = "short sleeve"
        else:
            sleeve_rolling_style = clothing_person_dict["sleeve rolling style"]

        clothing_prompt = clothing_template.format(
            category=clothing_dict['upper cloth category'],
            material=clothing_dict["material"],
            sleeve=clothing_dict["sleeve"],
            neckline=clothing_dict["neckline"],
        ).lower()

        full_prompt = full_template.format(
            gender=person_dict["gender"],
            body_shape=person_dict['body shape'],
            hair_length=person_dict['hair length'],
            pose=person_dict["pose"],
            hand_pose=person_dict['hand pose'],
            fit_of_clothing=person_dict['fit of upper cloth'],
            sleeve_rolling_style=sleeve_rolling_style,
            tucking_style=tucking_style,
            category=clothing_dict['upper cloth category'],
            material=clothing_dict["material"],
            neckline=clothing_dict["neckline"],
        ).lower()

        return full_prompt, clothing_prompt
    
    def generate(self, person_dict, clothing_dict, clothing_person_dict):
        full_prompt, clothing_prompt = self.create_prompt_v12(person_dict, clothing_dict, clothing_person_dict)
        return full_prompt, clothing_prompt

class IdentityTransform:
    def __call__(self, x):
        return x
    
def get_transform(txt_lst, **kwargs):
    trans_lst = []
    for tr in txt_lst:
        tr = tr.lower()
        if tr == "hflip":
            trans_lst.append(T.RandomHorizontalFlip())
        elif tr == "randomresizedcrop":
            trans_lst.append(T.RandomResizedCrop((kwargs["img_h"], kwargs["img_w"]), scale=(0.8, 1)))
        elif tr == "resize":
            trans_lst.append(T.Resize((kwargs["img_h"], kwargs["img_w"]), antialias=True))
        elif tr == "randomresizedcrop_dynamic":
            trans_lst.append(T.RandomResizedCrop((kwargs["img_h"], kwargs["img_w"]), scale=(0.5, 1), ratio=(0.3,2)))
        elif tr == "randomaffine":
            trans_lst.append(T.RandomAffine(degrees=0, translate=(0,0), scale=(0.8, 1.2)))
        elif tr == "randomaffine_dynamic":
            trans_lst.append(T.RandomAffine(degrees=(-30,30), translate=(0.1, 0.2), scale=(0.8, 1.2), fill=246))
        elif tr == "rotate":
            trans_lst.append(T.RandomAffine(degrees=(-30,30), fill=246))
        elif tr == "colorjitter":
            trans_lst.append(T.ColorJitter(
                brightness=(0.8,1.2),
                contrast=(0.8,1.2),
                saturation=(0.8,1.2),
                hue=(-0.1,0.1),
            ))
        elif tr == "colorjitter2":
            trans_lst.append(T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ))
        elif tr == "elastictransform":
            trans_lst.append(T.ElasticTransform())
        elif tr == "identity":
            trans_lst.append(IdentityTransform())
        else:
            raise NotImplementedError(tr)
    return trans_lst

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist


def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def get_mask_location(
        model_type, 
        category, 
        model_parse: Image.Image, 
        keypoint: dict, 
        width=384, height=512,
        radius=5,
        version=None,
        only_cloth=False,
        only_cloth_arm=False,
        only_cloth_armneck_with_dilate=False,
        densepose=None,
        use_pad=False,
):
    if category is None or category == "": category = "upper_body"
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)


    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)        
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        
        parser_mask_fixed_nolower = parser_mask_fixed.copy()

        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError(f"category {category}")
    
    if only_cloth:
        mask = Image.fromarray(parse_mask.astype(np.uint8) * 255)
        mask_gray = Image.fromarray(parse_mask.astype(np.uint8) * 127)
        if use_pad:
            dil_mask = binary_dilation(mask, iterations=30)
            mask = Image.fromarray(dil_mask.astype(np.uint8)*255)
            mask_gray = Image.fromarray(dil_mask.astype(np.uint8)*127)
        return mask, mask_gray
    elif only_cloth_arm:
        parse_mask = parse_mask + (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
        mask = Image.fromarray(parse_mask.astype(np.uint8) * 255)
        mask_gray = Image.fromarray(parse_mask.astype(np.uint8) * 127)
        return mask, mask_gray
    elif only_cloth_armneck_with_dilate:
        parse_mask = parse_mask + (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
        parse_mask = cv2.dilate(parse_mask.astype(np.uint8), np.ones((3,3), dtype=np.uint8), iterations=3)
        parse_mask = np.logical_or(parse_mask, (parse_array == 18).astype(np.float32))
        mask = Image.fromarray(parse_mask.astype(np.uint8) * 255)
        mask_gray = Image.fromarray(parse_mask.astype(np.uint8) * 127)
        return mask, mask_gray


    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)

        hip_right = np.multiply(tuple(pose_data[8][:2]), height / 512.0)
        hip_left = np.multiply(tuple(pose_data[11][:2]), height / 512.0)

        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

            
        if version == "v9": 
            pass
        else:
            hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
            hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
            hands = hands_left + hands_right
            parser_mask_fixed += hands
            if category == 'upper_body':
                parser_mask_fixed_nolower += hands

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    if category == 'upper_body':
        parser_mask_fixed_nolower = np.logical_or(parser_mask_fixed_nolower, parse_head)
    parse_mask = cv2.dilate(parse_mask, np.ones((radius, radius), np.uint16), iterations=5)
    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((radius, radius), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        
        if version == 'v7':
            arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=15)
        else:
            arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    if version not in ["v5", "v6"]:
        inpaint_mask = dst / 255 * 1

    if version is None or "official" in version:
        pass
    elif version == 'v9':
        inpaint_mask = dst
        hip_x1, hip_y1 = hip_left.astype(np.int64)
        hip_x2, hip_y2 = hip_right.astype(np.int64)
        inpaint_mask[hip_y1, hip_x1] = 255
        inpaint_mask[hip_y2, hip_x2] = 255
        coords = np.column_stack(np.where(inpaint_mask == 255))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        inpaint_mask[y_min:y_max+1, x_min:x_max+1] = 255
        inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(parser_mask_fixed_nolower))
        inpaint_mask = hole_fill(inpaint_mask.astype(np.uint8))
        inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(parse_head))
        inpaint_mask = inpaint_mask.astype(np.uint8)
    else:
        raise NotImplementedError(f"upper body version {version}")

        
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray