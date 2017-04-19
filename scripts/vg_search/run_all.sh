for i in {0..${N_JOBS}}; do
    export LOW=$((i*$INTERVAL)) HIGH=$(((i+1)*$INTERVAL-1)) GPUS=$((i / 2))
    zsh ./run.sh > ${SWEEP}.${LOW}_${HIGH}.log 2>&1 &
done
