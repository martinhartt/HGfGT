#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"
export AGIGA=data/agiga
export WORK=$ABS/working_agiga
export THREADS=30
export SCRIPTS=$ABS/dataset
export SCRIPTS_SUMMARY=$ABS/summary
export SPLITS=$ABS/$AGIGA
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
  shuf "$SPLITS/${SMALL}train.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/train.data.txt"
  shuf "$SPLITS/${SMALL}valid.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/valid.data.txt"
fi

set -x
mkdir -p $OUT_DIR

# Share the dictionary.
if [[ $* == *--filter* ]]
then
  UNK=5

  # Basic filtering on train/dev.
  python $SCRIPTS/filter.py $WORK/train.data.txt --firstSent 1 --wordOverlap 1 --lengthRange 1 > $WORK/train.filter.data.txt
  python $SCRIPTS/filter.py $WORK/valid.data.txt --firstSent 1 --wordOverlap 1 --lengthRange 1 > $WORK/valid.filter.data.txt

  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/train.filter.data.txt  $WORK/train.filter $UNK

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/train.filter.dict $OUT_DIR/filter.train.dict.torch

fi



if [[ $* == *--all* ]]
then
  UNK=5
  LIMIT=1000000

  # Basic filtering on train/dev.
  python $SCRIPTS/filter.py $WORK/train.data.txt --lengthRangeHeir 1 | head -n $LIMIT > $WORK/train.data.temp.txt
  python $SCRIPTS/filter.py $WORK/valid.data.txt --lengthRangeHeir 1 | head -n $LIMIT > $WORK/valid.data.temp.txt
  # python $SCRIPTS/filter.py $WORK/test.data.txt --lengthRangeHeir 1 | head -n $LIMIT > $WORK/test.data.temp.txt

  rm $WORK/train.data.txt $WORK/valid.data.txt

  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/train.data.temp.txt  $WORK/train.all $UNK

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/train.all.dict $OUT_DIR/all.train.dict.torch

fi


if [[ $* == *--test* ]]
then
  for SOURCE in AFP APW CNA NYT XIN
  do
    echo $SOURCE
    shuf "$SPLITS/${SMALL}test.splits" | grep $SOURCE
    shuf "$SPLITS/${SMALL}test.splits" | grep $SOURCE | xargs -I % bash -c "cat $WORK/raw/%" > $WORK/$SOURCE.test.data.txt

    python $SCRIPTS/filter.py $WORK/$SOURCE.test.data.txt --firstSent 1 --lengthRange 1 > $WORK/$SOURCE.test.filter.data.txt
    python $SCRIPTS/pull.py $WORK/$SOURCE.test.filter.data.txt $WORK/train.filter.dict

    python $SCRIPTS/filter.py $WORK/$SOURCE.test.data.txt --lengthRangeHeir 1 > $WORK/$SOURCE.test.all.data.txt
    python $SCRIPTS/pull.py $WORK/$SOURCE.test.all.data.txt $WORK/train.all.dict

    rm $WORK/$SOURCE.test.*data.txt
  done
fi
