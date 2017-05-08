#!/usr/kerberos/bin/zsh

source ~/anaconda/bin/activate jon-common
cd ~/scr/projects/pragmatic-tuna
export PYTHONPATH=.

BASE_MODEL=/tmp/vg.012.0074
logdir=$VG_LOG_DIR/<expnum:>
[[ ! -d $logdir ]] || (echo "logdir $logdir already exists"; exit)
cp -rp $BASE_MODEL $logdir || exit

# Substitute paths in checkpoint file
base_model_esc=`echo $BASE_MODEL | sed 's:/:\\\\/:g'`
logdir_esc=`echo $logdir | sed 's:/:\\\\/:g'`
sed -i "s/$base_model_esc/$logdir_esc/g" $logdir/checkpoint

python pragmatic_tuna/scripts/vg_rsa.py --corpus_path data/vg_processed_2_3.split_adv.dedup.pkl \
    --batch_size 64 --mode dream \
    --speaker_hidden_dim 565 --listener_learning_rate <llr:L:0.0001,0.01> --speaker_lr_factor <slr:L:0.1,500> --optimizer momentum \
    --fast_mapping_k <fm:E:0,64> --fast_mapping_threshold 0 \
    --n_dream_iters <nd:E:1001,2001> --dream_ratio <dr:L:0.4,0.6> --dream_stabilizer_factor <dsf:L:0.2,0.8> --dream_p_swap <dps:L:0.2,0.8> \
    --logdir $logdir > ${logdir}/stdout 2> ${logdir}/stderr
