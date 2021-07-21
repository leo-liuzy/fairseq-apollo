# Remember to set variable in this file correctly.
# Usage: ./preprocess_paras.sh "en-ru en-zh ...(separated by a single space)"
#

lg_pairs="ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
lg_count=1

for lg_pair in $lg_pairs; do
    echo "Language ${lg_count}"
    ./preprocess_para.sh $lg_pair
    echo
    lg_count=$( expr $lg_count + 1 )
done
echo "Done"
