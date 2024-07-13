#!/bin/bash

GT_image_path = 
generated_image_path=

python main.py \
--exp ffhq \
--real_dir "$GT_image_path" \
--gen_dir "$generated_image_path" \
--n_attr 20 \
--n_point 10000 \
--attr_type USER # USER, BLIP, color, shape, or.. (we only offer USER now)


