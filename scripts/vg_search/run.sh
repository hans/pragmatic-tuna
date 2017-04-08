#!/bin/zsh

host=`hostname | tr '.' '\n' | head -n1`
export VG_LOG_DIR=/$host/scr1/`whoami`/vg/$SWEEP
mkdir -p $VG_LOG_DIR

echo "Setting up for sweep: " $SWEEP
echo "Running trials from low to high:" $LOW $HIGH
echo "Using GPU:" $GPUS
echo "With logdir:" $VG_LOG_DIR

export CUDA_VISIBLE_DEVICES=$GPUS

for expt in {$LOW..$HIGH}; do
    echo $expt
    echo ${SWEEP}/$expt - `hostname` - $GPUS - at `git rev-parse HEAD` >> ~/tuna_assignments.txt

    script=${SWEEP}/template.$(printf "%04i" "$expt").sh
    bash $script
done
