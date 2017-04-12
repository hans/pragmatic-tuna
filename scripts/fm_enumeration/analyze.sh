for x in `ls results/*.out`; do grep 'L_ADV' $x | tail -n 5 > $x.dream; done
for x in `ls results/*.dream`; do fname=`echo $x | sed "s/out\.dream/dev_success/"`; awk -F "\t" '{split($3,a,":"); sum += a[2]; n+= 1} END {print sum / n}' < $x > $fname; done
for x in `ls results/*.dream`; do fname=`echo $x | sed "s/out\.dream/pt_success/"`; awk -F "\t" '{split($7,a,":"); sum += a[2]; n+= 1} END {print sum / n}' < $x > $fname; done
