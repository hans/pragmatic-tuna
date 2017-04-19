for i in {0..7}; do
    export LOW=$((i*125)) HIGH=$(((i+1)*125-1)) GPUS=$((i / 2))
    zsh ./run.sh > ${SWEEP}.${LOW}_${HIGH}.log 2>&1 &
done
