#!/bin/bash
#rm -rf logdir/train/*
        #--dim=896 \

#	--data_dir=./mix/out \

python test.py \
	--data_dir=../speech_mix/train \
	--test_dir=../speech_mix/test/audio1_3-audio2.wav \
	--silence_threshold=0.1 \
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

