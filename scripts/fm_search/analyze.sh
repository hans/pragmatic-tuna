#!/usr/bin/zsh

SWEEP_DIR=/jagupard15/scr1/jgauthie/vg_dream/006_fm

echo "K\tADVFM\tFM\tPT\tID"
for x in `find $SWEEP_DIR -type d`; do
    k=`awk -F ':' '/fast_mapping_k/ {gsub(",","",$2); print $2}' < $x/params`
    fm_avg=`grep ADVFM $x/stdout | tail -n5 | awk '{sum += $6; n += 1} END {print sum / n}'`
    advfm_avg=`grep ADVFM $x/stdout | tail -n5 | awk '{sum += $8; n += 1} END {print sum / n}'`
    dev_avg=`grep ADVFM $x/stdout | tail -n5 | awk '{sum += $10; n += 1} END {print sum / n}'`
    name=`basename $x`
    echo "$k\t$advfm_avg\t$fm_avg\t$dev_avg\t$name"
done | grep -v nan | sort -n
