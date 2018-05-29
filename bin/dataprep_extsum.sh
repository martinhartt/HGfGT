export ABS="$(dirname $(dirname $0))"
export WORK=$ABS/working_agiga
export THREADS=32
export SCRIPTS_SUMMARY=$ABS/summary

set -e
set -x

export ID=$SLURM_ARRAY_TASK_ID
export MAX=32

for TYPE in valid train
do
  echo "" > $WORK/split_${ID}-${MAX}.${TYPE}.all.data.txt
  awk -v n=$MAX "NR%n==$ID" $WORK/$TYPE.data.temp.txt | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS_SUMMARY/extractive.py \{\} $WORK/split_${ID}-${MAX}.${TYPE}.all.data.txt 2> /dev/null
done
