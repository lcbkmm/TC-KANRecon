#!/bin/sh
CUDA_VISIBLE_DEVICES="3" nohup python evaluation/controlnet_eval.py > train_sd.log 2>&1 &