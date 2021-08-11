#!/bin/bash
#SBATCH --partition=isi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=10:00:00

num_shards=5
./preprocess_mono.sh en $num_shards
# ./preprocess_mono.sh ru $num_shards
# ./preprocess_mono.sh id $num_shards
# ./preprocess_mono.sh vi $num_shards
# ./preprocess_mono.sh de $num_shards
# ./preprocess_mono.sh fi $num_shards
# ./preprocess_mono.sh ja $num_shards
# ./preprocess_mono.sh es $num_shards
# ./preprocess_mono.sh fr $num_shards
# ./preprocess_mono.sh ko $num_shards
# ./preprocess_mono.sh zh-Hans $num_shards
# ./preprocess_mono.sh bg $num_shards
# ./preprocess_mono.sh th $num_shards
# ./preprocess_mono.sh el $num_shards
# ./preprocess_mono.sh ar $num_shards
# ./preprocess_mono.sh tr $num_shards
# ./preprocess_mono.sh hi $num_shards
# ./preprocess_mono.sh bn $num_shards
# ./preprocess_mono.sh ur $num_shards
# ./preprocess_mono.sh te $num_shards
# ./preprocess_mono.sh sw $num_shards
