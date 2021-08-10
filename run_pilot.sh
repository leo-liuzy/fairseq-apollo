#!/bin/bash
python fairseq_cli/train.py --data data-bin/XLM_pilot_run_21Langs_debug \
    --langs en:ar:bg:bn:de:el:es:fi:fr:hi:id:ja:ko:ru:sw:te:th:tr:ur:vi:zh-Hans \
    --lang-pairs ar-en:bg-en:de-en:el-en:en-es:en-fr:en-hi:en-ru:en-sw:en-th:en-tr:en-ur:en-vi:en-zh \
    --use-mono-data \
    --use-mcl \
    --use-para-data \
    --use-tcl \
    --task xlm_xcl \
    --arch xlmr_xcl_base \
    --max-sentences 6 \
    --criterion xlm_xcl \
    --optimizer adam \
    --lr 0.0005 \
    --adam-betas "(0.9,0.98)" \
    --clip-norm 1.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --weight-decay 0.0001 \
    --seed 42 \
    --max-update 100 \
    --save-dir checkpoints/XLM_pilot_run_21Langs_debug \
    --log-interval 1 \
    --log-format json \
    --tensorboard-logdir checkpoints/XLM_pilot_run_21Langs_debug/log \
    --restore-file data/xlmr.base/model.pt \
    # --dropout 0.1 \
    # --model-parallel-size 2
    # --cpu
    # --dataset-impl "mmap" \

