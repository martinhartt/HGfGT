#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"


export MODEL=$ABS/working_agiga/models/$1
export LENGTH=15
export OUT_DIR=$ABS/working_agiga/processed


if [[ $* == *--hier* ]]
then
  HIER="--hier 1"
  FILTER="all"
else
  HIER=""
  FILTER="filter"
fi

if [[ $* == *--no-repeat* ]]
then
  NOREPEAT="--noRepeat 1"
else
  NOREPEAT=""
fi

if [[ $* == *--edu* ]]
then
  for SOURCE in CAE CPE FCE KET PET
  do
    INPUT=$ABS/working_edu/$SOURCE.${FILTER}.article.txt
    OUTPUT=$ABS/working_edu/$SOURCE.${FILTER}.title.txt
    echo "# Evaluating EDU $SOURCE"
    echo -e "\n\n"

    python $ABS/summary/generate.py \
      --model $MODEL \
      --inputf "$INPUT" \
      --outputf "$OUTPUT" \
      --length $LENGTH \
      $HIER \
      --workingDir  $OUT_DIR \
      $NOREPEAT \
      --dictionary $OUT_DIR/${FILTER}.train.dict.torch
  done
fi

if [[ $* == *--agiga* ]]
then
  for SOURCE in AFP APW CNA NYT XIN
  do
    INPUT=$ABS/working_agiga/$SOURCE.test.${FILTER}.article.txt
    OUTPUT=$ABS/working_agiga/$SOURCE.test.${FILTER}.title.txt
    echo "# Evaluating GIGAWORD $SOURCE"
    echo -e "\n\n"

    python $ABS/summary/generate.py \
      --model $MODEL \
      --inputf "$INPUT" \
      --outputf "$OUTPUT" \
      --length $LENGTH \
      $HIER \
      --workingDir  $OUT_DIR \
      $NOREPEAT \
      --dictionary $OUT_DIR/${FILTER}.train.dict.torch
  done
fi
