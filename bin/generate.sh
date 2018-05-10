#/bin/bash

export ABS="$(dirname $(dirname $0))"
export EXAMPLE=$ABS/working_agiga/test.article.filter.txt
export MODEL=$ABS/working_agiga/models/$1
export LENGTH=15
export OUT_DIR=$ABS/working_agiga/processed


python $ABS/summary/generate.py \
 -modelFilename $MODEL \
 -inputf "$EXAMPLE" \
 -length $LENGTH \
 -workingDir  $OUT_DIR \
 -filter 1 \
 # -blockRepeatWords
