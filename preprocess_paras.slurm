#!/bin/bash
#SBATCH --partition=isi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
num_shards=5
./preprocess_para.sh ar-en $num_shards
# ./preprocess_para.sh bg-en $num_shards
# ./preprocess_para.sh de-en $num_shards
# ./preprocess_para.sh el-en $num_shards
# ./preprocess_para.sh en-es $num_shards
# ./preprocess_para.sh en-fr $num_shards
# ./preprocess_para.sh en-hi $num_shards
# ./preprocess_para.sh en-ru $num_shards
# ./preprocess_para.sh en-sw $num_shards
# ./preprocess_para.sh en-th $num_shards
# ./preprocess_para.sh en-tr $num_shards
# ./preprocess_para.sh en-ur $num_shards
# ./preprocess_para.sh en-vi $num_shards
# ./preprocess_para.sh en-zh $num_shards
