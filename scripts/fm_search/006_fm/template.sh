#!/usr/kerberos/bin/zsh

source ~/anaconda/bin/activate jon-common
cd ~/scr/projects/pragmatic-tuna
export PYTHONPATH=.

logdir=$VG_LOG_DIR/<expnum:>
[[ ! -d $logdir ]] || (echo "logdir $logdir already exists"; exit)
cp -rp $BASE_MODEL $logdir || exit

python pragmatic_tuna/scripts/vg_rsa.py --corpus_path data/vg_processed.dedup.pkl \
    --batch_size 64 \
    --speaker_hidden_dim 652 --listener_learning_rate <llr:L:0.0001,0.01> --speaker_lr_factor <slr:L:1,500> --optimizer sgd \
    --fast_mapping_k <fm:E:0,64> --fast_mapping_threshold <fmt:L:0.8,5> --fast_mapping_neg_synth 3 \
    --n_dream_iters <nd:E:501,1001> --dream_ratio <dr:L:0.4,0.6> --dream_stabilizer_factor <dsf:L:0.3,0.7> --dream_p_swap <dps:L:0.2,0.8> \
    --logdir $logdir > ${logdir}/stdout 2> ${logdir}/stderr
