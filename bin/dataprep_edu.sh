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

GROUPS=find $WORK/raw/** -type d

# cat "$SPLITS/valid.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/valid.data.txt"

for g in GROUPS
do
  BASE=$(basename $g)
  cat $WORK/raw/$BASE/* > $WORK/$BASE.data.txt

  if [[ $* == *--filter* ]]
  then
    python $SCRIPTS/pull.py $WORK/$BASE.data.txt $AGIGA_WORK/train.filter.dict
  fi

  if [[ $* == *--all* ]]
  then
    python $SCRIPTS/pull.py $WORK/$BASE.data.txt $AGIGA_WORK/train.all.dict
  fi
done



rm -r $WORK/raw