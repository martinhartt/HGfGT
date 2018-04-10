#!/bin/bash

export ABS=/home/mh905/namas-python
export WORK=$ABS/working_agiga
export WINDOW=5
export OUT_DIR=$WORK/processed
export MDL_DIR=$WORK/models

# export LUA_PATH="$LUA_PATH;$ABS/?.lua"

#bash $ABS/prep_torch_data.sh $1

mkdir -p $MDL_DIR

python $ABS/summary-python/train.py -titleDir  $OUT_DIR/train/title/ \
 -articleDir  $OUT_DIR/train/article/ \
 -modelFilename  $MDL_DIR/$1 \
 -miniBatchSize  64 \
 -embeddingDim  64 \
 -bowDim  200 \
 -hiddenSize  64 \
 -epochs  20 \
 -learningRate 0.1 \
 -validArticleDir  $OUT_DIR/valid.filter/article/ \
 -validTitleDir  $OUT_DIR/valid.filter/title/ \
 -window  $WINDOW \
 -printEvery   100 \
 -encoderModel  "attenbow" \
 -attenPool  5 \
 -cuda 1
