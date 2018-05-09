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


if [[ $* == *--extract* ]]
then
  mkdir -p $WORK/raw
  find $ABS/$AGIGA/**/*.gz | wc -l | xargs echo "Total files to process:"
  # find $ABS/$AGIGA/**/*.gz | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_agiga.py \{\} $WORK

  # Compile the data into train/dev/test.
  cat "$SPLITS/train.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/train.data.txt"
  cat "$SPLITS/valid.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/valid.data.txt"
  cat "$SPLITS/test.splits"  | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/test.data.txt"
fi

set -x
mkdir -p $OUT_DIR

# Share the dictionary.
if [[ $* == *--filter* ]]
then
  # Basic filtering on train/dev.
  python $SCRIPTS/filter.py $WORK/train.data.txt > $WORK/train.data.filter.txt
  python $SCRIPTS/filter.py $WORK/valid.data.txt > $WORK/valid.data.filter.txt
  python $SCRIPTS/filter.py $WORK/test.data.txt > $WORK/test.data.filter.txt

  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/train.data.filter.txt  $WORK/train.filter $UNK

  # Split into title/article files.
  python $SCRIPTS/pull.py $WORK/test.data.filter.txt $WORK/train.filter.dict

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/train.filter.dict $OUT_DIR/filter.train.dict.torch

  python $SCRIPTS/build.py \
    -inputFile $WORK/train.data.filter.txt \
    -inDictionary $OUT_DIR/filter.train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "filter.train"

  python $SCRIPTS/build.py \
    -inputFile $WORK/valid.data.filter.txt \
    -inDictionary $OUT_DIR/filter.train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "filter.valid"

fi

if [[ $* == *--all* ]]
then
  # Compile dictionary.
  python $SCRIPTS/make_dict.py $WORK/train.data.txt  $WORK/train.all $UNK

  # Split into title/article files.
  python $SCRIPTS/pull.py $WORK/test.data.txt $WORK/train.all.dict

  # Constructing torch data files.
  python $SCRIPTS/build_dict.py $WORK/train.all.dict $OUT_DIR/all.train.dict.torch

  python $SCRIPTS/build.py \
    -inputFile $WORK/train.data.txt \
    -inDictionary $OUT_DIR/all.train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "all.train" \

  python $SCRIPTS/build.py \
    -inputFile $WORK/valid.data.txt \
    -inDictionary $OUT_DIR/all.train.dict.torch \
    -outDirectory $OUT_DIR \
    -outPrefix "all.valid" \

fi
