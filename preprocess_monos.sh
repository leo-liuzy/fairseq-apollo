# Remember to set variable in this file correctly.
# Usage: ./preprocess_mono.sh "en ru ...(separated by a single space)"
#

lgs="en ru id vi de fi ja es fr ko zh-Hans bg th el ar tr hi bn ur te sw"
DATA_DIR_ROOT="data"
BIN_DATA_DIR_ROOT="data-bin"
exp_name="XLM_pilot_run_21Langs_debug"
raw_input_dir="raw_${exp_name}"
tokenized_dir="tokenized_${exp_name}"
dict_path="${DATA_DIR_ROOT}/xlmr.base/dict.txt"
spm_path="${DATA_DIR_ROOT}/xlmr.base/sentencepiece.bpe.model"
task=mluna
corpus_type=monolingual

raw_mono_dir="${raw_input_dir}/${corpus_type}"
tokenized_mono_dir="${tokenized_dir}/${corpus_type}"
bin_destdir="${BIN_DATA_DIR_ROOT}/${exp_name}/${corpus_type}"

mkdir -p ${DATA_DIR_ROOT}/${tokenized_mono_dir}
mkdir -p "$bin_destdir"

lg_count=1

for lg in $lgs; do
    echo "Preprocessing ${lg} (Language ${lg_count})"

    echo "Tokenizing...."
    for split in train test valid; do
        echo "Tokenizing ${lg}.${split}"
        input=${DATA_DIR_ROOT}/${raw_mono_dir}/${split}.${lg}
        output=${DATA_DIR_ROOT}/${tokenized_mono_dir}/${split}.${lg}
        spm_encode --model=${spm_path} < "${input}" > "${output}"
    done

    echo "Binarizing...."
    fairseq-preprocess --task ${task} \
    --source-lang "${lg}" \
    --only-source --bpe sentencepiece \
    --trainpref "${DATA_DIR_ROOT}/${tokenized_mono_dir}/train" \
    --validpref "${DATA_DIR_ROOT}/${tokenized_mono_dir}/valid" \
    --testpref "${DATA_DIR_ROOT}/${tokenized_mono_dir}/test" \
    --srcdict $dict_path \
    --destdir $bin_destdir

    # Since we only have a source language, the output file has a None for the
    # target language. Remove this
    for stage in train test valid; do
      mv "$bin_destdir/$stage.$lg-None.$lg.bin" "$bin_destdir/$stage.$lg.bin"
      mv "$bin_destdir/$stage.$lg-None.$lg.idx" "$bin_destdir/$stage.$lg.idx"
    done
    # we use sentencepiece, so dict is shared
    mv "$bin_destdir/dict.${lg}.txt" "$bin_destdir/../dict.txt"
    echo
    lg_count=$( expr $lg_count + 1 )
done
