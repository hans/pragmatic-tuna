#!/usr/bin/zsh

source activate jon-common
export PYTHONPATH=.

rm -rf /tmp/tuna && python pragmatic_tuna/scripts/lot_rsa.py --corpus_path data/spatial_simple.json --gold_path data/spatial_simple_gold.json --debug \
     --fn_selection spatial_simple --max_timesteps 4 --atom_attribute shape \
     --learning_rate 0.01 --dream --num_listener_samples 128 --max_rejections_after_trial 4 --embedding_dim 32 \
     --num_runs 20 --num_trials 4 --speaker_model sequence

 # 2>&1 | less -R

# --gold_path data/exclusivity_simple_gold.json \
