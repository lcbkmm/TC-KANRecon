#!/bin/sh
CUDA_VISIBLE_DEVICES="0" nohup python stable_diffusion/zheer_sd.py > train_sd.log 2>&1 &