#!/bin/sh

. p pragmatic-tuna
cd ~/scr/projects/pragmatic-tuna
export PYTHONPATH=.

logdir=$VG_LOG_DIR/<expnum:>
mkdir -p $logdir

python pragmatic_tuna/scripts/vg_rsa.py --corpus_path data/vg_processed.pkl --n_iters 20000 --batch_size <batch_size:E:64,128> \
    --listener_learning_rate <llr:L:0.001,0.1> --speaker_lr_factor <slr:L:50,500> --dropout_keep_prob <drop:E:0.8,1.0> \
    --logdir $logdir > ${logdir}/stdout 2> ${logdir}/stderr
