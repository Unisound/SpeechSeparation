#!/bin/bash
#rm -rf logdir/train/*
        #--dim=896 \

#	--data_dir=./mix/out \

export CUDA_VISIBLE_DEVICES='3'
python train.py \
	--data_dir=/home/xuerq/data/librispeech/LibriSpeech/train-clean-100Bk/train \
	--test_dir=/home/xuerq/data/librispeech/LibriSpeech/train-clean-100Bk/test \
	--silence_threshold=0.1 \
	--sample_size=20488 \
        --big_frame_size=8 \
        --frame_size=2 \
        --q_levels=256 \
        --rnn_type=GRU \
        --dim=896 \
        --n_rnn=1 \
        --seq_len=256 \
        --emb_size=256 \
        --batch_size=1 \
	--optimizer=adam \
	--num_gpus=1

