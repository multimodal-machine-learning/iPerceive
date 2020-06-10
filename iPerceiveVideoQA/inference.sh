#!/usr/bin/env bash

python3 inference.py --input_streams sub vfeat dense \
    --seed 2018 --num_workers 4 --log_freq 2000 --bsz 16 --device_ids 0 1 --hsz 128 --max_dc_l 100 --input_encoder_n_heads 0 --cls_encoder_n_heads 0 #--mode test #--cs
