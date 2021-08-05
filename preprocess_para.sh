#!/bin/bash
# Remember to set variable in this file correctly.
# Usage: ./preprocess_mono.sh $lg_pair
#

lg_pair=$1
DATA_DIR_ROOT="data"
BIN_DATA_DIR_ROOT="data-bin"
exp_name="XLM_pilot_run_21Langs"
raw_input_dir="raw_${exp_name}"
tokenized_dir="tokenized_${exp_name}"
dict_path="${DATA_DIR_ROOT}/xlmr.base/dict.txt"
spm_path="${DATA_DIR_ROOT}/xlmr.base/sentencepiece.bpe.model"
task=xlmr_xcl
corpus_type=bilingual

raw_para_dir="${raw_input_dir}/${corpus_type}"
tokenized_para_dir="${tokenized_dir}/${corpus_type}"
bin_destdir="${BIN_DATA_DIR_ROOT}/${exp_name}/${corpus_type}"

mkdir -p ${DATA_DIR_ROOT}/${tokenized_para_dir}
mkdir -p "$bin_destdir"


echo "Preprocessing ${lg_pair}"

echo "Tokenizing...."
for split in train test valid; do
    echo "Tokenizing ${split}.${lg_pair}"
    for lg in $(echo $lg_pair | sed -e 's/\-/ /g'); do
      input=${DATA_DIR_ROOT}/${raw_para_dir}/${lg_pair}.${lg}.${split}
      output=${DATA_DIR_ROOT}/${tokenized_para_dir}/${split}.${lg_pair}.${lg}
      python spm_encode.py --model=${spm_path} < "${input}" > "${output}"
    done
done
IFS='-' read -ra LANGs <<< "$lg_pair"
source_lang=${LANGs[0]}
target_lang=${LANGs[1]}
echo "Binarizing...."
#echo $source_lang
#echo $target_lang
#echo $LANGs
#echo $bin_destdir
fairseq-preprocess --task ${task} \
--source-lang "${source_lang}" \
--target-lang "${target_lang}" \
--bpe sentencepiece \
--trainpref "${DATA_DIR_ROOT}/${tokenized_para_dir}/train.${lg_pair}" \
--validpref "${DATA_DIR_ROOT}/${tokenized_para_dir}/valid.${lg_pair}" \
--testpref "${DATA_DIR_ROOT}/${tokenized_para_dir}/test.${lg_pair}" \
--srcdict $dict_path \
--tgtdict $dict_path \
--destdir $bin_destdir

# we save bilingal data by language pair
mkdir -p "${bin_destdir}/$lg_pair"
#
## Since we only have a source language, the output file has a None for the
## target language. Remove this
for stage in train test valid; do
  for lg in $(echo $lg_pair | sed -e 's/\-/ /g'); do
    mv "$bin_destdir/$stage.$lg_pair.$lg.bin" "$bin_destdir/${lg_pair}/$stage.${lg}.bin"
    mv "$bin_destdir/$stage.$lg_pair.$lg.idx" "$bin_destdir/${lg_pair}/$stage.${lg}.idx"
  done
done
## we use sentencepiece, so dict is shared
for lg in $(echo $lg_pair | sed -e 's/\-/ /g'); do
  mv "$bin_destdir/dict.${lg}.txt" "${BIN_DATA_DIR_ROOT}/${exp_name}/dict.txt"
done
#echo "Done"
