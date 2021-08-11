#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=gpu
#SBATCH --job-name=xnli_xlmr_en
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=164g
#SBATCH --gres=gpu:v100:8
#SBATCH --cpus-per-task=16
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=36:00:00
#SBATCH --array=0

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}

# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

lr=0.0000075
per_gpu_batch_size=8
data_root=data
model_dir=xlmr.base.hg
num_epoch=30
max_len=256
lang=en
output_dir=lr${lr}_perGPUbsz${per_gpu_batch_size}_epoch${num_epoch}_maxlen${max_len}_lang${lang}_${model_dir}

mkdir -p ${output_dir}
cp $0 ${output_dir}/run.sh

# according to https://github.com/pytorch/fairseq/issues/2057#issuecomment-643674771
# no warmup is used and decay and we use same language for validation
python run_xnli.py \
  --model_name_or_path ${data_root}/xlmr.base.hg \
  --max_seq_length $max_len \
  --do_tarin \
  --do_eval \
  --learning_rate $lr \
  --per_device_train_batch_size $per_gpu_batch_size \
  --evaluation_strategy epoch \
  --num_train_epochs $num_epoch \
  --train_language $lang \
  --language $lang \
  --gradient_accumulation_steps 2 \
  --save_total_limit 2 \
  --output_dir $output_dir
