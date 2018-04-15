
export ABS=/Users/martinhartt/uni/project/namas

export EXAMPLE=$ABS/working_edu/train.article.txt
export MODEL=$ABS/working_agiga/models/ok
export LENGTH=15
export OUT_DIR=$ABS/working_agiga/processed


python $ABS/summary/run.py \
 -modelFilename $MODEL \
 -inputf "$EXAMPLE" \
 -length $LENGTH \
 -titleDir $OUT_DIR/train/title/ \
 -articleDir $OUT_DIR/train/article/ \
 # -blockRepeatWords
