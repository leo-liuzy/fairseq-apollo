#
# Usage: ./tokenize_spm.sh
#

lgs="en ru id vi de fi ja es fr ko zh-Hans bg th el ar tr hi bn ur te sw"
input_dir="data/raw_XLM_pilot_run_21Langs_debug"
output_dir="data/XLM_pilot_run_21Langs_debug"
spm_path="data/xlmr.base/sentencepiece.bpe.model"

lg_count=1

for lg in $lgs; do
    echo "Language ${lg_count}"
    for split in train test valid; do
        echo "Tokenizing ${lg}.${split}"
        input=${input_dir}/${split}.${lg}
        output=${output_dir}/${split}.${lg}
        spm_encode --model=${spm_path} < $input > $output
    done
    lg_count=$( expr $lg_count + 1 )
    echo
done
