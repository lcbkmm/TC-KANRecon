#!/bin/sh
CUDA_VISIBLE_DEVICES="2" nohup python stable_diffusion/zheer_late_controlnet.py > train_diff_loss.log 2>&1 &