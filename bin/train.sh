#!/bin/bash

set -e
set -x

export ABS="$(dirname $(dirname $0))"
export WORK=$ABS/working_agiga
export WINDOW=5
export OUT_DIR=$WORK/processed
export MDL_DIR=$WORK/models

# export LUA_PATH="$LUA_PATH;$ABS/?.lua"

#bash $ABS/prep_torch_data.sh $1

mkdir -p $MDL_DIR

if [[ $* == *--heir* ]]
then
  HEIR="-heir 1"
  TYPE=".all"
else
  HEIR=""
  TYPE=".filter"
fi

if [[ $* == *--restore* ]]
then
  RESTORE="-restore 1"
else
  RESTORE=""
fi

if [[ $* == *--glove* ]]
then
  GLOVE="-glove 1"
else
  GLOVE=""
fi

date
python $ABS/summary/train.py \
  -modelFilename  $MDL_DIR/$1 \
  -trainFile $WORK/train${TYPE}.data.txt \
  -validFile $WORK/valid${TYPE}.data.txt \
  -dictionary $OUT_DIR/all.train.dict.torch \
  -batchSize  64 \
  -bowDim  300 \
  -hiddenSize  64 \
  -epochs  15 \
  -learningRate 0.01 \
  -window  $WINDOW \
  -printEvery   100 \
  -attenPool  5 \
  $HEIR \
  $GLOVE \
  $RESTORE
date


# curl "https://maker.ifttt.com/trigger/ping/with/key/bZk7rWKnuJhYlSHus2DL5L"

# cd $ABS/jobs
# sbatch train
