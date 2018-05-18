#/bin/bash
set -e

export ABS="$(dirname $(dirname $0))"


export MODEL=$ABS/working_agiga/models/$1
export LENGTH=15
export OUT_DIR=$ABS/working_agiga/processed


if [[ $* == *--edu* ]]
then
  for g in CAE CPE FCE KET PET
  do
    INPUT=$ABS/working_edu/$g.article.filter.txt
    OUTPUT=$ABS/working_edu/$g.title.filter.txt
    echo "# Evaluating $g"

    python $ABS/summary/generate.py \
      -modelFilename $MODEL \
      -inputf "$INPUT" \
      -outputf "$OUTPUT" \
      -length $LENGTH \
      -workingDir  $OUT_DIR 2> /dev/null
  done
fi

if [[ $* == *--agiga* ]]
then
  INPUT=$ABS/working_agiga/test.article.filter.txt
  OUTPUT=$ABS/working_agiga/test.title.filter.txt

  echo "# Evaluating Gigaword"
  python $ABS/summary/generate.py \
    -modelFilename $MODEL \
    -inputf "$INPUT" \
    -outputf "$OUTPUT" \
    -length $LENGTH \
    -workingDir  $OUT_DIR 2> /dev/null
fi
