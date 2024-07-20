#!/bin/sh
CUDA_VISIBLE_DEVICES="0" nohup python my_vqvae/train_cvae.py > train.log 2>&1 &