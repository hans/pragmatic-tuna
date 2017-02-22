#!/usr/bin/zsh

source activate jon-common
export PYTHONPATH=.

rm -rf /tmp/tuna && python pragmatic_tuna/scripts/lot_rsa.py --corpus_path data/spatial_complex.json --gold_path data/spatial_complex_gold.json --debug \
     --fn_selection spatial_complex --max_timesteps 4 --atom_attribute type --dream --dream_samples 10 \
     --learning_rate 0.01 --num_listener_samples 128 --max_rejections_after_trial 4 --embedding_dim 8 \
     --num_runs 5 --num_trials 6 --speaker_model sequence

 # 2>&1 | less -R

# --gold_path data/exclusivity_simple_gold.json \
