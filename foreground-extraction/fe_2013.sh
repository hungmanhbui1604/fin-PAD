#!/bin/bash

data_dir="/home/hmb1604/datasets/LivDet/2013"
out_dir="/home/hmb1604/datasets/LivDet/foreground_2013"
sensors=("Biometrika" "Italdata")
block_sizes=(3 3)
deltas=(15 2)
kernel_sizes=(9 9)

for i in "${!sensors[@]}"; do
    python foreground_extraction.py -i "${data_dir}/${sensors[i]}" -o "$out_dir/${sensors[i]}" -b ${block_sizes[i]} -d ${deltas[i]} -k ${kernel_sizes[i]}
done