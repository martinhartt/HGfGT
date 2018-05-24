#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"


export MODEL=$ABS/working_agiga/models/$1
export LENGTH=15
export OUT_DIR=$ABS/working_agiga/processed


if [[ $* == *--heir* ]]
then
  HEIR="--heir 1"
  FILTER="all"
else
  HEIR=""
  FILTER="filter"
fi


if [[ $* == *--edu* ]]
then
  for g in CAE CPE FCE KET PET
  do
    INPUT=$ABS/working_edu/$g.${FILTER}.article.txt
    OUTPUT=$ABS/working_edu/$g.${FILTER}.title.txt
    echo "# Evaluating $g"
    echo -e "\n\n"

    python $ABS/summary/generate.py \
      --model $MODEL \
      --inputf "$INPUT" \
      --outputf "$OUTPUT" \
      --length $LENGTH \
      $HEIR \
      --workingDir  $OUT_DIR \
      --dictionary $OUT_DIR/${FILTER}.train.dict.torch
  done
fi

if [[ $* == *--agiga* ]]
then
  INPUT=$ABS/working_agiga/test.${FILTER}.article.txt
  OUTPUT=$ABS/working_agiga/test.${FILTER}.title.txt

  echo "# Evaluating Gigaword"
  echo -e "\n\n"

  python $ABS/summary/generate.py \
    --model $MODEL \
    --inputf "$INPUT" \
    --outputf "$OUTPUT" \
    --length $LENGTH \
    $HEIR \
    --workingDir  $OUT_DIR \
    --dictionary $OUT_DIR/${FILTER}.train.dict.torch
fi
