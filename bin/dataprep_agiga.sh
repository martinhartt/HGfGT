#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"
export AGIGA=data/agiga
export WORK=$ABS/working_agiga
export THREADS=30
export SCRIPTS=$ABS/dataset
export SCRIPTS_SUMMARY=$ABS/summary
export SPLITS=$ABS/$AGIGA
export UNK=5
export OUT_DIR=$WORK/processed


# Construct the title-article pairs from gigaword

export SMALL=""
if [[ $* == *--small* ]]
then
  export SMALL="small_"
fi

if [[ $* == *--extract* ]]
then
  mkdir -p $WORK/raw
  find $ABS/$AGIGA/**/*.gz | wc -l | xargs echo "Total files to process:"
  find $ABS/$AGIGA/**/*.gz | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_agiga.py \{\} $WORK

fi

if [[ $* == *--splits* ]]
then
  # Compile the data into train/dev/test.
  cat "$SPLITS/${SMALL}train.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/train.data.txt"
  cat "$SPLITS/${SMALL}valid.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/valid.data.txt"
  cat "$SPLITS/${SMALL}test.splits"  | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/test.data.txt"
fi

set -x
mkdir -p $OUT_DIR

# Share the dictionary.
if [[ $* == *--filter* ]]
then
  # Basic filtering on train/dev.
  python $SCRIPTS/filter.py $WORK/train.data.txt --firstSent 1 --wordOverlap 1 --lengthRange 1 > $WORK/train.filter.data.txt
  python $SCRIPTS/filter.py $WORK/valid.data.txt --firstSent 1 --wordOverlap 1 --lengthRange 1 > $WORK/valid.filter.data.txt
  python $SCRIPTS/filter.py $WORK/test.data.txt --firstSent 1 --wordOverlap 1 --lengthRange 1 > $WORK/test.filter.data.txt

  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/train.filter.data.txt  $WORK/train.filter $UNK

  # Split into title/article files.
  python $SCRIPTS/pull.py $WORK/test.filter.data.txt $WORK/train.filter.dict

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/train.filter.dict $OUT_DIR/filter.train.dict.torch

fi

if [[ $* == *--all* ]]
then
  # Basic filtering on train/dev.
  python $SCRIPTS/filter.py $WORK/train.data.txt > $WORK/train.data.temp.txt
  python $SCRIPTS/filter.py $WORK/valid.data.txt > $WORK/valid.data.temp.txt
  python $SCRIPTS/filter.py $WORK/test.data.txt > $WORK/test.data.temp.txt

  # HACK Reduced the dataset size as it is too large
  cat $WORK/train.data.temp.txt | python $SCRIPTS_SUMMARY/extractive.py > $WORK/train.all.data.txt
  cat $WORK/test.data.temp.txt | python $SCRIPTS_SUMMARY/extractive.py > $WORK/test.all.data.txt
  cat $WORK/valid.data.temp.txt | python $SCRIPTS_SUMMARY/extractive.py > $WORK/valid.all.data.txt

  rm $WORK/*temp*

  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/train.all.data.txt  $WORK/train.all $UNK

  # Split into title/article files.
  python $SCRIPTS/pull.py $WORK/test.all.data.txt $WORK/train.all.dict

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/train.all.dict $OUT_DIR/all.train.dict.torch

fi
