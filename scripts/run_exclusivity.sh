#!/usr/bin/zsh

source activate jon-common
export PYTHONPATH=.

rm -rf /tmp/tuna && python pragmatic_tuna/scripts/lot_rsa.py --corpus_path data/exclusivity_simple.json \
     --fn_selection simple --num_trials 2 --atom_attribute shape --learning_rate 0.01 --dream \
     --num_listener_samples 128 --max_rejections_after_trial 2 --num_runs 20 --embedding_dim 8 --debug \
     --speaker_model sequence

 # 2>&1 | less -R

# --gold_path data/exclusivity_simple_gold.json \
