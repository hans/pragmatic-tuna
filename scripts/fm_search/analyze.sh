#!/usr/bin/zsh

alias -g .strip="| sed 's/^[ \t]\+//; s/[ \t]\+$//'"
alias -g .avg='| awk "{sum += \$1; n+= 1} END {print sum / n}"'
SWEEP_DIR=/jagupard13/scr1/jgauthie/vg_dream/D013

echo "K\tADVFM_AVG\tADVFM_in\tADVFM_on\tADVFM_near\tADVFM_behind\tADVFM_under\tFM\tPT\tID"
for x in `find $SWEEP_DIR -type d`; do
    k=`awk -F ':' '/fast_mapping_k/ {gsub(",","",$2); print $2}' < $x/params`
    fm_avg=`grep L_FM $x/stdout | tail -n3 | awk '{sum += $6; n += 1} END {print sum / n}'`
    dev_avg=`grep L_FM $x/stdout | tail -n3 | awk '{sum += $8; n += 1} END {print sum / n}'`

    # # ADVFM results are on separate lines, blocked by relation type
    # advfm_avgs=`grep -P 'in\t\s+\d+' $x/stdout | tail -n1 | sed 's/[ \t]\+/\t/g' | sed 's/^[ \t]\+|\n$//g'`
    in_avg=`grep -Po 'in\s+([\d.]+)(\t|$)' $x/stdout | tail -n3 | cut -f2 .strip .avg`
    on_avg=`grep -Po 'on\s+([\d.]+)(\t|$)' $x/stdout | tail -n3 | cut -f2 .strip .avg`
    near_avg=`grep -Po 'near\s+([\d.]+)(\t|$)' $x/stdout | tail -n3 | cut -f2 .strip .avg`
    behind_avg=`grep -Po 'behind\s+([\d.]+)(\t|$)' $x/stdout | tail -n3 | cut -f2 .strip .avg`
    under_avg=`grep -Po 'under\s+([\d.]+)(\t|$)' $x/stdout | tail -n3 | cut -f2 .strip .avg`

    advfm_avg=`echo "$in_avg\t$on_avg\t$near_avg\t$behind_avg\t$under_avg" | awk '{print ($1+$2+$3+$4+$5)/5;}'`

    name=`basename $x`
    echo "$k\t$advfm_avg\t$in_avg\t$on_avg\t$near_avg\t$behind_avg\t$under_avg\t$fm_avg\t$dev_avg\t$name"
done | grep -v nan | sort -n
