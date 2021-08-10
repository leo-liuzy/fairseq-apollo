# Remember to set variable in this file correctly.
# Usage: ./preprocess_monos.sh "en ru ...(separated by a single space)"
#

lgs="en ru id vi de fi ja es fr ko zh-Hans bg th el ar tr hi bn ur te sw"
# lgs=$1
# "fr ko zh-Hans bg th"
# lgs="el ar tr hi bn ur te sw"
# lgs="id vi de fi ja es fr ko zh-Hans bg th el ar tr hi bn ur te sw"
lg_count=1

for lg in $lgs; do
    echo "Language ${lg_count}"
    ./preprocess_mono.sh $lg
    echo
    lg_count=$( expr $lg_count + 1 )
done
echo "Done"
