#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"
export EDU=data/edu
export AGIGA_WORK=$ABS/working_agiga
export WORK=$ABS/working_edu
export THREADS=4
export SCRIPTS=$ABS/dataset
export SPLITS=$ABS/$EDU
export UNK=1

# Step 1: Construct the title-article pairs from gigaword
mkdir -p $WORK/raw
find $ABS/$EDU/**/*.txt | wc -l | xargs echo "Total files to process:"
find $ABS/$EDU/**/*.txt | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_edu.py \{\} $WORK

for g in CAE CPE FCE KET PET
do
  echo $g
  BASE=$(basename $WORK/raw/$g)
  cat $WORK/raw/$BASE/* > $WORK/$BASE.data.txt

  if [[ $* == *--filter* ]]
  then
    python $SCRIPTS/filter.py $WORK/$BASE.data.txt --firstSent 1 --lengthRange 1 > $WORK/$BASE.data.filter.txt
    python $SCRIPTS/pull.py $WORK/$BASE.data.filter.txt $AGIGA_WORK/train.filter.dict
  fi

  if [[ $* == *--all* ]]
  then
    python $SCRIPTS/filter.py $WORK/$BASE.data.txt > $WORK/$BASE.all.data.txt
    python $SCRIPTS/pull.py $WORK/$BASE.all.data.txt $AGIGA_WORK/train.all.dict
  fi
done


rm $WORK/*data*
rm -r $WORK/raw
