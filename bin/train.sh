#!/bin/bash

export ABS="$(dirname $(dirname $0))"
export WORK=$ABS/working_agiga
export WINDOW=5
export OUT_DIR=$WORK/processed
export MDL_DIR=$WORK/models

# export LUA_PATH="$LUA_PATH;$ABS/?.lua"

#bash $ABS/prep_torch_data.sh $1

mkdir -p $MDL_DIR

if [[ $* == *--rush* ]]
then
  HEIR=0
else
  HEIR=1
fi

date
python $ABS/summary/train.py \
  -workingDir  $OUT_DIR \
  -modelFilename  $MDL_DIR/$1 \
  -miniBatchSize  64 \
  -embeddingDim  64 \
  -bowDim  200 \
  -hiddenSize  64 \
  -epochs  15 \
  -learningRate 0.01 \
  -window  $WINDOW \
  -printEvery   100 \
  -encoderModel  "attenbow" \
  -attenPool  5 \
  -heir $HEIR
date


curl "https://maker.ifttt.com/trigger/ping/with/key/bZk7rWKnuJhYlSHus2DL5L"

cd $ABS/jobs
sbatch train
