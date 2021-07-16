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
mkdir -p ${debug_para_dir}

echo "Monolingual......"
# Create debug corpus for monolingual data
for i in train,5000 test,1000 valid,1000; do IFS=","; set -- $i;
    split=$1
    num_example=$2
    echo "Create ${split} split:"
    for f in ${mono_dir}/*.${split}; do
        LG="${f%.$split}"  # trim extension
        LG="${LG#${mono_dir}/}" # trim path to source_dir
        echo "Processing ${LG}"
        head -n $num_example $f > ${debug_mono_dir}/${split}.${LG}
    done
    echo
done

echo "Bilingual......"
# Create debug corpus for bilingual data
for i in train,5000 test,1000 valid,1000; do IFS=","; set -- $i;
    # for i in valid,1000; do IFS=","; set -- $i;
    split=$1
    num_example=$2
    echo "Create ${split} split:"
    for f in ${para_dir}/*.${split}; do
        LG="${f%.$split}"  # trim extension
        LG="${LG#${para_dir}/}" # trim path to source_dir
        echo "Processing ${LG}"
        head -n $num_example $f > ${debug_para_dir}/${split}.${LG}
    done
    echo
done
