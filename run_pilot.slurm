#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=gpu
#SBATCH --job-name=XLM_pilot_run_21Langs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=164g
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
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
lr=0.0005
max_sentences=6
num_update=52000
base_exp="XLM_pilot_run_21Langs_XLM+XCL"
exp_name="bszPerGPU${max_sentences}_lr${lr}_update${num_update}_${base_exp}"

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

python fairseq_cli/train.py --data ${DATA} \
    --langs en:ar:bg:bn:de:el:es:fi:fr:hi:id:ja:ko:ru:sw:te:th:tr:ur:vi:zh-Hans \
    --lang-pairs ar-en:bg-en:de-en:el-en:en-es:en-fr:en-hi:en-ru:en-sw:en-th:en-tr:en-ur:en-vi:en-zh \
    --use-mono-data \
    --use-mcl \
    --use-para-data \
    --use-tcl \
    --task xlm_xcl \
    --arch xlmr_xcl_base \
    --max-sentences ${max_sentences} \
    --criterion xlm_xcl \
    --optimizer adam \
    --lr ${lr} \
    --adam-betas "(0.9,0.98)" \
    --clip-norm 1.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --weight-decay 0.0001 \
    --seed 42 \
    --max-update ${num_update} \
    --save-dir ${SAVE} \
    --log-interval 20 \
    --log-format json \
    --tensorboard-logdir ${SAVE}/log \
    --restore-file data/xlmr.base/model.pt
