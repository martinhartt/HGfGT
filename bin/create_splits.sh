#/bin/bash

export ABS="$(dirname $(dirname $0))"
DATASET=$1

python $ABS/dataset/splits.py $DATASET
