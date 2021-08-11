#!/bin/bash
# Remember to set variable in this file correctly.
# Usage: ./preprocess_mono.sh $lg_pair
#

lg_pair=$1
NUM_SHARDS=$2
DATA_DIR_ROOT="data"
BIN_DATA_DIR_ROOT="data-bin"
exp_name="XLM_pilot_run_21Langs"
raw_input_dir="raw_${exp_name}"
tokenized_dir="tokenized_${exp_name}"
dict_path="${DATA_DIR_ROOT}/xlmr.base/dict.txt"
spm_path="${DATA_DIR_ROOT}/xlmr.base/sentencepiece.bpe.model"
task=xlm_xcl
corpus_type=bilingual

raw_para_dir="${raw_input_dir}/${corpus_type}"
tokenized_para_dir="${tokenized_dir}/${corpus_type}"
bin_destdir="${BIN_DATA_DIR_ROOT}/${exp_name}/${corpus_type}"

mkdir -p ${DATA_DIR_ROOT}/${tokenized_para_dir}
mkdir -p "$bin_destdir"

tokenized=1
echo "Preprocessing ${lg_pair}"

echo "Tokenizing...."
for split in train test valid; do
    echo "Tokenizing ${split}.${lg_pair}"
    for lg in $(echo $lg_pair | sed -e 's/\-/ /g'); do
      # input=${DATA_DIR_ROOT}/${raw_para_dir}/${lg_pair}.${lg}.${split}
      input=${DATA_DIR_ROOT}/${raw_para_dir}/${split}.${lg_pair}.${lg}
      output=${DATA_DIR_ROOT}/${tokenized_para_dir}/${split}.${lg_pair}.${lg}
      if [ $tokenized -ne 1 ]
      then
        python scripts/spm_encode.py --model=${spm_path} < "${input}" > "${output}"
      fi

      # optionally shard the dataset
      if [ $NUM_SHARDS -gt 1 ]
      then
        echo "Sharding...."
#        python scripts/shard_docs.py ${output} --num-shards $NUM_SHARDS
        echo $((`wc -l < ${output}` / $NUM_SHARDS))
        echo ${output}
        split -l$((`wc -l < ${output}` / $NUM_SHARDS)) ${output} ${output}.shard -da 1
        for i in $(seq 0 $((NUM_SHARDS-1)) );
        do
          shard_source_dir=${DATA_DIR_ROOT}${i}/${tokenized_para_dir}
          mkdir -p $shard_source_dir
          mv "${output}.shard${i}" ${shard_source_dir}/${split}.${lg_pair}.${lg}
        done
      fi

    done
done
IFS='-' read -ra LANGs <<< "$lg_pair"
source_lang=${LANGs[0]}
target_lang=${LANGs[1]}
echo "Binarizing...."
if [ $NUM_SHARDS -le 1 ]
then
  fairseq-preprocess --task ${task} \
  --source-lang "${source_lang}" \
  --target-lang "${target_lang}" \
  --bpe     sentencepiece \
  --trainpref "${DATA_DIR_ROOT}/${tokenized_para_dir}/train.${lg_pair}" \
  --validpref "${DATA_DIR_ROOT}/${tokenized_para_dir}/valid.${lg_pair}" \
  --testpref "${DATA_DIR_ROOT}/${tokenized_para_dir}/test.${lg_pair}" \
  --srcdict $dict_path \
  --tgtdict $dict_path \
  --destdir $bin_destdir \
  --bpe sentencepiece

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
else
  for i in $(seq 0 $((NUM_SHARDS-1)) );
  do
    shard_source_dir=${DATA_DIR_ROOT}${i}/${tokenized_para_dir}
    shard_bin_destdir="${BIN_DATA_DIR_ROOT}${i}/${exp_name}/${corpus_type}"
    mkdir -p shard_bin_destdir

    fairseq-preprocess --task ${task} \
      --source-lang "${source_lang}" \
      --target-lang "${target_lang}" \
      --bpe sentencepiece \
      --trainpref "${shard_source_dir}/train.${lg_pair}" \
      --validpref "${shard_source_dir}/valid.${lg_pair}" \
      --testpref "${shard_source_dir}/test.${lg_pair}" \
      --srcdict $dict_path \
      --tgtdict $dict_path \
      --destdir $shard_bin_destdir \
      --bpe sentencepiece

      # we save bilingal data by language pair
      mkdir -p "${shard_bin_destdir}/$lg_pair"
      #
      ## Since we only have a source language, the output file has a None for the
      ## target language. Remove this
      for stage in train test valid; do
        for lg in $(echo $lg_pair | sed -e 's/\-/ /g'); do
          mv "$shard_bin_destdir/$stage.$lg_pair.$lg.bin" "$shard_bin_destdir/${lg_pair}/$stage.${lg}.bin"
          mv "$shard_bin_destdir/$stage.$lg_pair.$lg.idx" "$shard_bin_destdir/${lg_pair}/$stage.${lg}.idx"
        done
      done
      ## we use sentencepiece, so dict is shared
      for lg in $(echo $lg_pair | sed -e 's/\-/ /g'); do
        mv "$shard_bin_destdir/dict.${lg}.txt" "${BIN_DATA_DIR_ROOT}${i}/${exp_name}/dict.txt"
      done
  done
fi
echo "Done"
