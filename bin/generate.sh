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
    EXAMPLE=$ABS/working_edu/$g.article.filter.txt
    echo "Evaluating edu $g"
    if [[ $* == *--greedy* ]]
    then
      python $ABS/summary/generate.py \
      -modelFilename $MODEL \
      -inputf "$EXAMPLE" \
      -length $LENGTH \
      -workingDir  $OUT_DIR \
      -greedy 1
    else
      python $ABS/summary/generate.py \
      -modelFilename $MODEL \
      -inputf "$EXAMPLE" \
      -length $LENGTH \
      -workingDir  $OUT_DIR
    fi
  done
fi

if [[ $* == *--agiga* ]]
then
  EXAMPLE=$ABS/working_agiga/test.article.filter.txt

  echo "Evaluating Gigaword"
  if [[ $* == *--greedy* ]]
  then
    python $ABS/summary/generate.py \
    -modelFilename $MODEL \
    -inputf "$EXAMPLE" \
    -length $LENGTH \
    -workingDir  $OUT_DIR \
    -greedy 1
  else
    python $ABS/summary/generate.py \
    -modelFilename $MODEL \
    -inputf "$EXAMPLE" \
    -length $LENGTH \
    -workingDir  $OUT_DIR
  fi
fi
