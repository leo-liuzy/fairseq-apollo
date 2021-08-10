#!/bin/bash
#SBATCH --partition=isi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=2:00:00

# ./preprocess_para.sh ar-en
# ./preprocess_para.sh bg-en
# ./preprocess_para.sh de-en
# ./preprocess_para.sh el-en
# ./preprocess_para.sh en-es
# ./preprocess_para.sh en-fr
# ./preprocess_para.sh en-hi
# ./preprocess_para.sh en-ru
# ./preprocess_para.sh en-sw
# ./preprocess_para.sh en-th
# ./preprocess_para.sh en-tr
# ./preprocess_para.sh en-ur
# ./preprocess_para.sh en-vi
#./preprocess_para.sh en-zh
echo $1