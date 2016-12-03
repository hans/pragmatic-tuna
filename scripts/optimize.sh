#!/bin/bash

for LR in "0.001" "0.01" "0.1" "0.5" "1"
do
    for MR in 0 1 2 3 4 5
    do
        for NS in 5 10 15 20
        do
            for SM in "discrete" "window"
            do
                echo "Learning rate: $LR"
                echo "Max rejections: $MR"
                echo "Number of listener samples: $NS"
                echo "Speaker model $SM"
                rm -rf /tmp/tuna && python pragmatic_tuna/models/lot_rsa.py --corpus_path data/spatial_super_simple.json --atom_attribute shape --learning_method xent --num_listener_samples $NS --num_trials 2 --learning_rate $LR  --gold_path data/spatial_super_simple_gold.json --num_runs 10 --max_rejections_after_trial $MR --speaker_model $SM | grep "Average accuracy" 
            done
        done
    done
done