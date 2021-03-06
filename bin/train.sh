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

if [[ $* == *--hier* ]]
then
  HIER="--hier 1"
  TYPE="all."
  LR="0.01"
elif [[ $* == *--mix* ]]
then
  HIER="--hier 1"
  TYPE="filter."
  LR="0.01"
else
  HIER=""
  TYPE="filter."
  LR="0.1"
fi

if [[ $* == *--pooling* ]]
then
  POOL="--pooling 1"
else
  POOL=""
fi

if [[ $* == *--extra* ]]
then
  EXTRA="--extraAttnLinear 1"
else
  EXTRA=""
fi

if [[ $* == *--restore* ]]
then
  RESTORE="--restore 1"
else
  RESTORE=""
fi

if [[ $* == *--glove* ]]
then
  GLOVE="--glove 1"
else
  GLOVE=""
fi

if [[ $* == *--no-attn* ]]
then
  NOATTN="--noAttn 1"
else
  NOATTN=""
fi

date
python $ABS/summary/train.py \
  --model  $MDL_DIR/$1 \
  --train $WORK/train.${TYPE}data.txt \
  --valid $WORK/valid.${TYPE}data.txt \
  --dictionary $OUT_DIR/${TYPE}train.dict.torch \
  --batchSize  64 \
  --bowDim  300 \
  --learningRate $LR \
  --window  $WINDOW \
  --printEvery   100 \
  --attenPool  5 \
  --useTeacherForcing 1 \
  $HIER \
  $GLOVE \
  $RESTORE \
  $NOATTN \
  $EXTRA \
  $POOL
date


# curl "https://maker.ifttt.com/trigger/ping/with/key/bZk7rWKnuJhYlSHus2DL5L"

# cd $ABS/jobs
# sbatch train
