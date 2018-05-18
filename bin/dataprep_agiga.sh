#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"
export AGIGA=data/agiga
export WORK=$ABS/working_agiga
export THREADS=30
export SCRIPTS=$ABS/dataset
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
  cat "$SPLITS/${SMALL}train.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/${SMALL}train.data.txt"
  cat "$SPLITS/${SMALL}valid.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/${SMALL}valid.data.txt"
  cat "$SPLITS/${SMALL}test.splits"  | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/${SMALL}test.data.txt"
fi

set -x
mkdir -p $OUT_DIR

# Share the dictionary.
if [[ $* == *--filter* ]]
then
  # Basic filtering on train/dev.
  python $SCRIPTS/filter.py $WORK/${SMALL}train.data.txt > $WORK/${SMALL}train.data.filter.txt
  python $SCRIPTS/filter.py $WORK/${SMALL}valid.data.txt > $WORK/${SMALL}valid.data.filter.txt
  python $SCRIPTS/filter.py $WORK/${SMALL}test.data.txt > $WORK/${SMALL}test.data.filter.txt

  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/${SMALL}train.data.filter.txt  $WORK/${SMALL}train.filter $UNK

  # Split into title/article files.
  python $SCRIPTS/pull.py $WORK/${SMALL}test.data.filter.txt $WORK/${SMALL}train.filter.dict

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/${SMALL}train.filter.dict $OUT_DIR/${SMALL}filter.train.dict.torch

  python $SCRIPTS/build.py \
    -inputFile $WORK/${SMALL}train.data.filter.txt \
    -inDictionary $OUT_DIR/filter.${SMALL}train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "filter.${SMALL}train"

  python $SCRIPTS/build.py \
    -inputFile $WORK/${SMALL}valid.data.filter.txt \
    -inDictionary $OUT_DIR/filter.${SMALL}train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "filter.${SMALL}valid"

fi

if [[ $* == *--all* ]]
then
  # Basic filtering on train/dev.
  python $SCRIPTS/filter_lengths.py $WORK/${SMALL}train.data.txt > $WORK/${SMALL}train.data.temp.txt
  python $SCRIPTS/filter_lengths.py $WORK/${SMALL}valid.data.txt > $WORK/${SMALL}valid.data.temp.txt
  python $SCRIPTS/filter_lengths.py $WORK/${SMALL}test.data.txt > $WORK/${SMALL}test.data.temp.txt

  # HACK Reduced the dataset size as it is too large
  shuf -n 300000 $WORK/${SMALL}train.data.temp.txt | python $SCRIPTS/extractive.py > $WORK/${SMALL}train.all.data.txt
  shuf -n 2000 $WORK/${SMALL}test.data.temp.txt | python $SCRIPTS/extractive.py > $WORK/${SMALL}test.all.data.txt
  shuf -n 2000 $WORK/${SMALL}valid.data.temp.txt | python $SCRIPTS/extractive.py > $WORK/${SMALL}valid.all.data.txt

  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/${SMALL}train.all.data.txt  $WORK/${SMALL}train.all $UNK

  # Split into title/article files.
  python $SCRIPTS/pull.py $WORK/${SMALL}test.all.data.txt $WORK/${SMALL}train.all.dict

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/${SMALL}train.all.dict $OUT_DIR/all.${SMALL}train.dict.torch

  python $SCRIPTS/build.py \
    -inputFile $WORK/${SMALL}train.all.data.txt \
    -inDictionary $OUT_DIR/all.${SMALL}train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "all.${SMALL}train" \

  python $SCRIPTS/build.py \
    -inputFile $WORK/${SMALL}valid.all.data.txt \
    -inDictionary $OUT_DIR/all.${SMALL}train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "all.${SMALL}valid" \

fi
