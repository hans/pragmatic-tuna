#!/usr/bin/zsh

alias -g .strip="| sed 's/^[ \t]\+//; s/[ \t]\+$//'"
SWEEP_DIR=/jagupard15/scr1/jgauthie/vg_dream/011

echo "K\tADVFM_AVG\tADVFM_in\tADVFM_on\tADVFM_near\tADVFM_behind\tFM\tPT\tID"
for x in `find $SWEEP_DIR -type d`; do
    k=`awk -F ':' '/fast_mapping_k/ {gsub(",","",$2); print $2}' < $x/params`
    fm_avg=`grep L_FM $x/stdout | tail -n1 | awk '{sum += $6; n += 1} END {print sum / n}'`
    dev_avg=`grep L_FM $x/stdout | tail -n1 | awk '{sum += $8; n += 1} END {print sum / n}'`

    # # ADVFM results are on separate lines, blocked by relation type
    # advfm_avgs=`grep -P 'in\t\s+\d+' $x/stdout | tail -n1 | sed 's/[ \t]\+/\t/g' | sed 's/^[ \t]\+|\n$//g'`
    in_avg=`grep -Po 'in\s+([\d.]+)' $x/stdout | tail -n1 | cut -f2 .strip`
    on_avg=`grep -Po 'on\s+([\d.]+)' $x/stdout | tail -n1 | cut -f2 .strip`
    near_avg=`grep -Po 'near\s+([\d.]+)' $x/stdout | tail -n1 | cut -f2 .strip`
    behind_avg=`grep -Po 'behind\s+([\d.]+)' $x/stdout | tail -n1 | cut -f2 .strip`

    advfm_avg=`echo "$in_avg\t$on_avg\t$near_avg\t$behind_avg" | awk '{print ($1+$2+$3+$4)/4;}'`

    name=`basename $x`
    echo "$k\t$advfm_avg\t$in_avg\t$on_avg\t$near_avg\t$behind_avg\t$fm_avg\t$dev_avg\t$name"
done | grep -v nan | sort -n
