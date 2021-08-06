#!/bin/bash
#SBATCH --partition=isi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00


# ./preprocess_mono.sh en
# ./preprocess_mono.sh ru
# ./preprocess_mono.sh id
# ./preprocess_mono.sh vi
# ./preprocess_mono.sh de
# ./preprocess_mono.sh fi
# ./preprocess_mono.sh ja
./preprocess_mono.sh es

# ./preprocess_mono.sh fr
# ./preprocess_mono.sh ko
# ./preprocess_mono.sh zh-Hans
# ./preprocess_mono.sh bg
# ./preprocess_mono.sh th
# ./preprocess_mono.sh el
# ./preprocess_mono.sh ar
# ./preprocess_mono.sh tr
# ./preprocess_mono.sh hi
# ./preprocess_mono.sh bn
# ./preprocess_mono.sh ur
# ./preprocess_mono.sh te
# ./preprocess_mono.sh sw
