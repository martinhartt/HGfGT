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

echo "Step 1: Construct the title-article pairs from gigaword"
mkdir -p $WORK/raw
find $ABS/$EDU/**/*.txt | wc -l | xargs echo "Total files to process:"
find $ABS/$EDU/**/*.txt | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_edu.py \{\} $WORK


echo "Step 2: Compile the data into train/dev/test."
cat "$SPLITS/train.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/train.data.txt"
cat "$SPLITS/valid.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/valid.data.txt"
cat "$SPLITS/test.splits"  | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/test.data.txt"


echo "Step 3: Construct title-article files."
python $SCRIPTS/pull.py $WORK/train.data.txt $AGIGA_WORK/train.title.dict $AGIGA_WORK/train.article.dict
python $SCRIPTS/pull.py $WORK/valid.data.txt $AGIGA_WORK/train.title.dict $AGIGA_WORK/train.article.dict
python $SCRIPTS/pull.py $WORK/test.data.txt $AGIGA_WORK/train.title.dict $AGIGA_WORK/train.article.dict


rm -r $WORK/raw
