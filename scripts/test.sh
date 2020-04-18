#!/usr/bin/env sh

python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py \
            ./my_configs/retinanet_r50_fpn_1x.py     \
            ./xxx.pth   \
            --launcher pytorch
