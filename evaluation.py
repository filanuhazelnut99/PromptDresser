from glob import glob
import os
from os.path import join as opj
import argparse

from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from cleanfid import fid
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class PairedDataset(Dataset):
    def __init__(self, pred_ps, gt_ps, img_h, img_w):
        self.pred_ps = pred_ps
        self.gt_ps = gt_ps
        self.transform = T.Compose([
            T.Resize((img_h, img_w)),
            T.ToTensor(),
        ])
        len(self.pred_ps) == len(self.gt_ps)
    def __len__(self):
        return len(self.pred_ps)
    def __getitem__(self, idx):
        pred_img = self.transform(Image.open(self.pred_ps[idx]).convert("RGB"))
        gt_img = self.transform(Image.open(self.gt_ps[idx]).convert("RGB"))
        return pred_img, gt_img
    
@torch.no_grad()
def get_metrics(pred_dir, gt_dir, img_h, img_w, is_unpaired):
    pred_ps = sorted(glob(opj(pred_dir, "*.jpg"))) + sorted(glob(opj(pred_dir, "*.png")))  + sorted(glob(opj(pred_dir, "*.jpeg")))
    gt_ps = sorted(glob(opj(gt_dir, "*.jpg"))) + sorted(glob(opj(gt_dir, "*.png")))  + sorted(glob(opj(gt_dir, "*.jpeg")))
    if not is_unpaired:
        assert len(pred_ps) == len(gt_ps), f"in paired setting, # of pred and gt should be equal : {len(pred_ps)} vs {len(gt_ps)}"
    print(f"# of pred_paths : {len(pred_ps)}, # of gt paths : {len(gt_ps)}, {'unpaired' if is_unpaired else 'paired'}, img_h : {img_h}, img_w : {img_w}")
    if is_unpaired:   
        avg_ssim = 0.0
        avg_lpips = 0.0
    else:     
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()

        paired_dataset = PairedDataset(pred_ps, gt_ps, img_h, img_w)
        paired_loader = DataLoader(paired_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        for pred, gt in tqdm(paired_loader, total=len(paired_loader), desc="Calculating SSIM and LPIPS"):
            pred = pred.cuda()
            gt = gt.cuda()
            
            ssim.update(pred, gt)
            lpips.update(pred, gt)
        
        avg_ssim = ssim.compute().item()
        avg_lpips = lpips.compute().item()
    fid_score = fid.compute_fid(pred_dir, gt_dir, mode="clean", use_dataparallel=False, dataset_split="custom")
    kid_score = fid.compute_kid(pred_dir, gt_dir, mode="clean", use_dataparallel=False, dataset_split="custom")

    return avg_ssim, avg_lpips, fid_score, kid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--img_h", type=int, default=1024)
    parser.add_argument("--img_w", type=int, default=768)
    parser.add_argument("--category", default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--pair_type", default=None)
    args = parser.parse_args()


    args.pred_dir = args.pred_dir.rstrip("/")
    if args.pair_type is None:
        pair_type = args.pred_dir.split("/")[-1]
    else:
        pair_type = args.pair_type
    if args.category is None:
        category = args.pred_dir.split("/")[-2]
    else:
        category = args.category
    gt_dir = "./DATA/zalando-hd-resized/test_fine/image"
        
    ssim_score, lpips_score, fid_score, kid_score = get_metrics(
        args.pred_dir, gt_dir, 
        args.img_h, args.img_w, 
        is_unpaired=pair_type == "unpaired"
    )
    print(f"ssim : {ssim_score}, lpips : {lpips_score}, fid : {fid_score}, kid : {kid_score}")

    save_path = opj(os.path.dirname(args.pred_dir), f"{pair_type}_results_{args.img_h}.txt")
    with open(save_path, "w") as f:
        f.write(f"ssim : {ssim_score}\n")
        f.write(f"lpips : {lpips_score}\n")
        f.write(f"fid : {fid_score}\n")
        f.write(f"kid_score : {kid_score}")
    print(f"file save to {save_path}")
    
    

    
           