lgs="en ru id vi de fi ja es fr ko zh-Hans bg th el ar tr hi bn ur te sw"
DATA_DIR_ROOT="data"
BIN_DATA_DIR_ROOT="data-bin"
raw_input_dir="${DATA_DIR_ROOT}/raw_XLM_pilot_run_21Langs_debug"
output_dir="${DATA_DIR_ROOT}/XLM_pilot_run_21Langs_debug"
monolingual_dir="XLM_pilot_run_21Langs_debug/monolingual"
tokenized_dir="${DATA_DIR_ROOT}/${monolingual_dir}"
dict="${DATA_DIR_ROOT}/xlmr.base/dict.txt"
destdir="${BIN_DATA_DIR_ROOT}/${monolingual_dir}"
task=mluna
spm_path="${DATA_DIR_ROOT}/xlmr.base/sentencepiece.bpe.model"


mkdir -p "$destdir"

for lg in $lgs; do
    echo "Preprocessing ${lg}"
    fairseq-preprocess --task ${task} \
    --source-lang ${lg} \
    --only-source --bpe sentencepiece \
    --trainpref ${tokenized_dir}/train \
    --validpref ${tokenized_dir}/valid \
    --testpref ${tokenized_dir}/test \
    --srcdict $dict \
    --destdir $destdir

    # Since we only have a source language, the output file has a None for the
    # target language. Remove this
    for stage in train test valid; do
      mv "$destdir/$stage.$lg-None.$lg.bin" "$destdir/$stage.$lg.bin"
      mv "$destdir/$stage.$lg-None.$lg.idx" "$destdir/$stage.$lg.idx"
    done
    # we use sentencepiece, so dict is shared
    mv "$destdir/dict.${lg}.txt" "$destdir/../dict.txt"
    echo
done
