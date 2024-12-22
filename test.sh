CUDA_VISIBLE_DEVICES=0 python inference.py \
 --config_p "./configs/VITONHD.yaml" \
 --pretrained_unet_path "./checkpoints/VITONHD/model/pytorch_model.bin" \
 --save_name VITONHD

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 --num_processes 4 --multi_gpu --main_process_port 4466 inference.py \
 --config_p "./configs/VITONHD.yaml" \
 --pretrained_unet_path "./checkpoints/VITONHD/model/pytorch_model.bin" \
 --save_name VITONHD