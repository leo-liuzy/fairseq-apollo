data_dir_root="data"
source_dir="${data_dir_root}/raw_XLM_pilot_run_21Langs"
# debug_dir="XLM_pilot_run_21Langs_debug"
debug_dir="${source_dir}_debug"
echo ${debug_dir}

mono_dir="${source_dir}/monolingual"
para_dir="${source_dir}/bilingual"

debug_mono_dir="${debug_dir}/monolingual"
debug_para_dir="${debug_dir}/bilingual"

mkdir -p ${debug_dir}
mkdir -p ${debug_mono_dir}

# Create debug corpus for monolingual data

N_TEST=1000
split=test
echo "Create ${split} split:"
for f in ${mono_dir}/*.${split}; do
    LG="${f%.$split}"  # trim extension
    LG="${LG#${mono_dir}/}" # trim path to source_dir
    echo "Processing ${LG}"
    head -n $N_TEST $f > ${debug_mono_dir}/${split}.${LG}
done
echo

N_VALID=1000
split=valid
echo "Create ${split} split:"
for f in ${mono_dir}/*.${split}; do
    LG="${f%.$split}"  # trim extension
    LG="${LG#${mono_dir}/}" # trim path to source_dir
    echo "Processing ${LG}"
    head -n $N_VALID $f > ${debug_mono_dir}/${split}.${LG}
done
echo

N_TRAIN=5000
split=train
echo "Create ${split} split:"
for f in ${mono_dir}/*.${split}; do
    LG="${f%.$split}"  # trim extension
    LG="${LG#${mono_dir}/}" # trim path to source_dir
    echo "Processing ${LG}"
    head -n $N_TRAIN $f > ${debug_mono_dir}/${split}.${LG}
done

