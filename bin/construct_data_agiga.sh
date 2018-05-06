#/bin/bash
set -e


export ABS="$(dirname $(dirname $0))"
export AGIGA=data/agiga
export WORK=$ABS/working_agiga
export THREADS=4
export SCRIPTS=$ABS/dataset
export SPLITS=$ABS/$AGIGA
export UNK=5

echo "Step 1: Construct the title-article pairs from gigaword"
mkdir -p $WORK/raw
find $ABS/$AGIGA/**/*.gz | wc -l | xargs echo "Total files to process:"
find $ABS/$AGIGA/**/*.gz | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_agiga.py \{\} $WORK


echo "Step 2: Compile the data into train/dev/test."
cat "$SPLITS/train.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/train.data.txt"
cat "$SPLITS/valid.splits" | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/valid.data.txt"
cat "$SPLITS/test.splits"  | xargs -I % bash -c "cat $WORK/raw/%" > "$WORK/test.data.txt"

echo "Step 3: Basic filtering on train/dev."
python $SCRIPTS/filter.py $WORK/train.data.txt > $WORK/train.data.filter.txt
python $SCRIPTS/filter.py $WORK/valid.data.txt > $WORK/valid.data.filter.txt
python $SCRIPTS/filter.py $WORK/test.data.txt > $WORK/test.data.filter.txt


echo "Step 4: Compile dictionary."
python $SCRIPTS/make_dict.py $WORK/train.data.filter.txt  $WORK/train.filter $UNK
python $SCRIPTS/make_dict.py $WORK/train.data.txt  $WORK/train $UNK


echo "Step 5: Construct title-article files."
python $SCRIPTS/pull.py $WORK/train.data.filter.txt $WORK/train.filter.title.dict $WORK/train.filter.article.dict
python $SCRIPTS/pull.py $WORK/valid.data.filter.txt $WORK/train.filter.title.dict $WORK/train.filter.article.dict
python $SCRIPTS/pull.py $WORK/test.data.filter.txt $WORK/train.filter.title.dict $WORK/train.filter.article.dict

python $SCRIPTS/pull.py $WORK/train.data.txt $WORK/train.title.dict $WORK/train.article.dict
python $SCRIPTS/pull.py $WORK/valid.data.txt $WORK/train.title.dict $WORK/train.article.dict
python $SCRIPTS/pull.py $WORK/test.data.txt $WORK/train.title.dict $WORK/train.article.dict

rm $WORK/*data*.txt
rm -r $WORK/raw

echo "Step 6: Constructing torch data files."
# bash $ABS/bin/prep_torch_data.sh $WORK/

rm -r $WORK/raw
