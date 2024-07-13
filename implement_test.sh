#!/bin/bash

llist=("prgan_50000_256_no_a")


for element in "${llist[@]}"; do
    echo $element
    python main.py \
    --exp ffhq \
    --real_dir "/workspace/dataset/dongkyun/PROJECTS/entangle_fid/for_rebuttal/sample_size/real/50000_256_no_a" \
    --gen_dir "/workspace/dataset/dongkyun/PROJECTS/entangle_fid/experiment_ffhq/$element" \
    --n_attr 20 \
    --n_point 10000 \
    --attr_type USER # USER, BLIP, color, shape, or..
done



