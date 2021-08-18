#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=gpu
#SBATCH --job-name=XLM_pilot_run_21Langs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=164g
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=8
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=24:00:00
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

#DATE=`date +%Y%m%d`
PROJ_DIR=/home1/zliu9986/fairseq-apollo
SAVE_ROOT=${PROJ_DIR}/checkpoints
DATA=${PROJ_DIR}/data-bin/XLM_pilot_run_21Langs
lr=1e-6
max_sentences=4
update_freq=8
world_size=2
num_update=5200
base_exp="XLM_pilot_run_21Langs_MLM"
exp_name="bszPerGPU$((max_sentences * world_size * update_freq))_lr${lr}_update${num_update}_${base_exp}"

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

srun --label python fairseq_cli/train.py --data ${DATA} \
    --langs en:ar:bg:bn:de:el:es:fi:fr:hi:id:ja:ko:ru:sw:te:th:tr:ur:vi:zh-Hans \
    --lang-pairs ar-en:bg-en:de-en:el-en:en-es:en-fr:en-hi:en-ru:en-sw:en-th:en-tr:en-ur:en-vi:en-zh \
    --task xlm_xcl \
    --arch xlmr_xcl_base \
    --max-sentences ${max_sentences} \
    --criterion xlm_xcl \
    --optimizer adam \
    --lr ${lr} \
    --adam-betas "(0.9,0.98)" \
    --clip-norm 1.0 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 4000 \
    --weight-decay 0.0001 \
    --seed 42 \
    --max-update ${num_update} \
    --update-freq ${update_freq} \
    --save-dir ${SAVE} \
    --log-interval 20 \
    --log-format json \
    --tensorboard-logdir logs/${SAVE} \
    --restore-file data/xlmr.base/model.pt \
    --distributed-port 3154 \
    --distributed-world-size $world_size \
    --use-mono-data \
    # --use-mlm \
    --use-mcl
    # --use-tcl \
    # --use-para-data \
    # --use-mlm \
    # --use-tlm 
