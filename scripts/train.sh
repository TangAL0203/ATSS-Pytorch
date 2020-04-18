#!/usr/bin/env sh

# train retinanet 1x baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py \
    ./my_configs/retinanet_r50_fpn_1x.py \
    --launcher pytorch \
    --work_dir ./work_dirs/ \
    --seed 1    \
    --validate  \


# train retinanet 1x with atss
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py \
    ./my_configs/retinanet_r50_fpn_atss_1x.py \
    --launcher pytorch \
    --work_dir ./work_dirs/ \
    --seed 2    \
    --validate  \
