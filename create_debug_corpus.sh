data_dir_root="data"
source_dir="${data_dir_root}/raw_XLM_pilot_run_21Langs"
# debug_dir="XLM_pilot_run_21Langs_debug"
debug_dir="${source_dir}_debug"

mono_dir="${source_dir}/monolingual"
para_dir="${source_dir}/bilingual"

debug_mono_dir="${debug_dir}/monolingual"
debug_para_dir="${debug_dir}/bilingual"

mkdir -p ${debug_dir}
mkdir -p ${debug_mono_dir}

# Create debug corpus for monolingual data
cp ${mono_dir}/*.test ${debug_mono_dir}/
cp ${mono_dir}/*.valid ${debug_mono_dir}/

N_TRAIN=50000
for f in ${mono_dir}/*.train; do
    LG="${f%.train}"  # trim extension
    LG="${LG#${mono_dir}/}" # trim path to source_dir
    echo "Processing ${LG}"
    head -n $N_TRAIN $f > ${debug_mono_dir}/${LG}.train
done

