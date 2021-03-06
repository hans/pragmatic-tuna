#!/usr/kerberos/bin/zsh

source ~/anaconda/bin/activate jon-common
cd ~/scr/projects/pragmatic-tuna
export PYTHONPATH=.

logdir=$VG_LOG_DIR/<expnum:>
mkdir -p $logdir

python pragmatic_tuna/scripts/vg_rsa.py --corpus_path data/vg_processed_2_3.split_adv.dedup.pkl --n_iters 20000 --mode pretrain \
    --batch_size <batch_size:E:64> \
    --listener_learning_rate <llr:L:0.0001,0.001> --speaker_lr_factor <slr:L:1,500> --dropout_keep_prob <drop:E:0.8,1.0> \
    --speaker_hidden_dim <shid:LI:256,1024> \
    --logdir $logdir > ${logdir}/stdout 2> ${logdir}/stderr
