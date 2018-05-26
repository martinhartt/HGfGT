export ABS="$(dirname $(dirname $0))"
export WORK=$ABS/working_agiga
export THREADS=30
export SCRIPTS_SUMMARY=$ABS/summary

set -e
set -x

for TYPE in valid train
do
  cat $WORK/split_*-*.${TYPE}.all.data.txt > $WORK/${TYPE}.all.data.txt
  rm $WORK/split_*-*.${TYPE}.all.data.txt
done
