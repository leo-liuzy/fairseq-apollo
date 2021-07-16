source_dir="data/XLM_pilot_run_21Langs_debug"

for split in train test valid; do
    echo "Split: ${split}"
    for f in ${source_dir}/*.${split}; do
        lg="${f%.$split}"  # trim extension
        lg="${lg#${source_dir}/}" # trim path to source_dir
        echo "Processing ${lg}"
        mv $f ${source_dir}/${split}.${lg}
        # head -n $N_TRAIN $f > ${debug_dir}/${LG}.train
    done
    echo
done
