#/bin/bash
set -e
set -x

export ABS="$(dirname $(dirname $0))"


export MODEL=$ABS/working_agiga/models/$1
export LENGTH=15
export OUT_DIR=$ABS/working_agiga/processed


if [[ $* == *--rush* ]]
then
  HEIR=""
else
  HEIR="-heir"
fi


if [[ $* == *--edu* ]]
then
  for g in CAE CPE FCE KET PET
  do
    INPUT=$ABS/working_edu/$g.article.filter.txt
    OUTPUT=$ABS/working_edu/$g.title.filter.txt
    echo "# Evaluating $g"
    echo -e "\n\n"

    python $ABS/summary/generate.py \
      -modelFilename $MODEL \
      -inputf "$INPUT" \
      -outputf "$OUTPUT" \
      -length $LENGTH \
      -heir $HEIR \
      -workingDir  $OUT_DIR
  done
fi

if [[ $* == *--agiga* ]]
then
  INPUT=$ABS/working_agiga/test.article.filter.txt
  OUTPUT=$ABS/working_agiga/test.title.filter.txt

  echo "# Evaluating Gigaword"
  echo -e "\n\n"

  python $ABS/summary/generate.py \
    -modelFilename $MODEL \
    -inputf "$INPUT" \
    -outputf "$OUTPUT" \
    -length $LENGTH \
    $HEIR \
    -workingDir  $OUT_DIR
fi
