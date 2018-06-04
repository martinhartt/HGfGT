#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"
export AGIGA=data/agiga
export EDU=data/edu
export WORK=$ABS/working_lm
export THREADS=4
export SCRIPTS=$ABS/dataset
export SCRIPTS_SUMMARY=$ABS/summary
export SPLITS=$ABS/$AGIGA
export UNK=5
export OUT_DIR=$WORK/processed


# Construct the title-article pairs from gigaword

if [[ $* == *--extract* ]]
then
  mkdir -p $WORK/raw
  find $ABS/$AGIGA/**/*.gz | wc -l | xargs echo "Total files to process:"
  find $ABS/$AGIGA/**/*.gz | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_agiga.py \{\} $WORK

  find $ABS/$EDU/**/*.txt | wc -l | xargs echo "Total files to process:"
  find $ABS/$EDU/**/*.txt | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_edu.py \{\} $WORK
fi

if [[ $* == *--splits* ]]
then
  # Compile the data into train/dev/test.
  # find $WORK/raw/**/*.txt | xargs -I % bash -c "cat %" > "$WORK/train.txt"
  cat $WORK/train.txt | tr '\t' '\n' | sed 's/<sb>/\
/g' > $WORK/train_final.txt
  cat $WORK/train_final.txt | tr ' ' '\n' | uniq > $WORK/train.vocab.txt
fi
