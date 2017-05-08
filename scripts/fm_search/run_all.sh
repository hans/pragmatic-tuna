for i in {0..7}; do
    export LOW=$((i*$INT)) HIGH=$(((i+1)*$INT-1)) GPUS=$((i / 2))
    zsh ./run.sh > ${SWEEP}.${LOW}_${HIGH}.log 2>&1 &
done
