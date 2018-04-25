#!/bin/bash

export ABS="$(dirname $(dirname $0))"

bash $ABS/bin/train.sh $1
bash $ABS/bin/eval.sh $1
