# Remember to set variable in this file correctly.
# Usage: ./preprocess_mono.sh $lg
#

lg=$1
NUM_SHARDS=$2
DATA_DIR_ROOT="data"
BIN_DATA_DIR_ROOT="data-bin"
exp_name="XLM_pilot_run_21Langs"
raw_input_dir="raw_${exp_name}"
tokenized_dir="tokenized_${exp_name}"
dict_path="${DATA_DIR_ROOT}/xlmr.base/dict.txt"
spm_path="${DATA_DIR_ROOT}/xlmr.base/sentencepiece.bpe.model"
task=xlm_xcl
corpus_type=monolingual

raw_mono_dir="${raw_input_dir}/${corpus_type}"
tokenized_mono_dir="${tokenized_dir}/${corpus_type}"
bin_destdir="${BIN_DATA_DIR_ROOT}/${exp_name}/${corpus_type}"

mkdir -p ${DATA_DIR_ROOT}/${tokenized_mono_dir}
mkdir -p "$bin_destdir"

tokenized=1
echo "Preprocessing ${lg}"

echo "Tokenizing...."
for split in train test valid; do
    echo "Tokenizing ${lg}.${split}"
    input=${DATA_DIR_ROOT}/${raw_mono_dir}/${split}.${lg}
    # input=${DATA_DIR_ROOT}/${raw_mono_dir}/${lg}.${split}
    output=${DATA_DIR_ROOT}/${tokenized_mono_dir}/${split}.${lg}
    if [ $tokenized -ne 1 ]
    then
      python spm_encode.py --model=${spm_path} < "${input}" > "${output}"
    fi

    # optionally shard the dataset
    if [ $NUM_SHARDS -gt 1 ]
    then
      python scripts/shard_docs.py ${output} --num-shards $NUM_SHARDS
      for i in $(seq 0 $((NUM_SHARDS-1)) );
      do
        shard_source_dir=${DATA_DIR_ROOT}${i}/${tokenized_mono_dir}
        mkdir -p $shard_source_dir
        mv "${output}.shard${i}" ${shard_source_dir}/${split}.${lg}
      done
    fi

done

echo "Binarizing...."
if [ $NUM_SHARDS -le 1 ]
then
  fairseq-preprocess --task ${task} \
  --source-lang "${lg}" \
  --only-source --bpe sentencepiece \
  --trainpref "${DATA_DIR_ROOT}/${tokenized_mono_dir}/train" \
  --validpref "${DATA_DIR_ROOT}/${tokenized_mono_dir}/valid" \
  --testpref "${DATA_DIR_ROOT}/${tokenized_mono_dir}/test" \
  --srcdict $dict_path \
  --destdir $bin_destdir \
  --bpe sentencepiece

  # we save monolingal data one lang by one lang
  mkdir -p "${bin_destdir}/$lg"

  # Since we only have a source language, the output file has a None for the
  # target language. Remove this
  for stage in train test valid; do
    mv "$bin_destdir/$stage.$lg-None.$lg.bin" "$bin_destdir/${lg}/$stage.bin"
    mv "$bin_destdir/$stage.$lg-None.$lg.idx" "$bin_destdir/${lg}/$stage.idx"
  done
  # we use sentencepiece, so dict is shared
  mv "$bin_destdir/dict.${lg}.txt" "$bin_destdir/dict.txt"
else
  for i in $(seq 0 $((NUM_SHARDS-1)) );
  do
    shard_source_dir=${DATA_DIR_ROOT}${i}/${tokenized_mono_dir}
    shard_bin_destdir="${BIN_DATA_DIR_ROOT}${i}/${exp_name}/${corpus_type}"
    mkdir -p shard_bin_destdir

    fairseq-preprocess --task ${task} \
    --source-lang "${lg}" \
    --only-source --bpe sentencepiece \
    --trainpref "${shard_source_dir}/train" \
    --validpref "${shard_source_dir}/valid" \
    --testpref "${shard_source_dir}/test" \
    --srcdict $dict_path \
    --destdir $shard_bin_destdir \
    --bpe sentencepiece

    # we save monolingal data one lang by one lang
    mkdir -p "${shard_bin_destdir}/$lg"

    # Since we only have a source language, the output file has a None for the
    # target language. Remove this
    for stage in train test valid; do
      mv "$shard_bin_destdir/$stage.$lg-None.$lg.bin" "$shard_bin_destdir/${lg}/$stage.bin"
      mv "$shard_bin_destdir/$stage.$lg-None.$lg.idx" "$shard_bin_destdir/${lg}/$stage.idx"
    done
    # we use sentencepiece, so dict is shared
    mv "$shard_bin_destdir/dict.${lg}.txt" "$shard_bin_destdir/dict.txt"
  done
fi

echo "Done"
