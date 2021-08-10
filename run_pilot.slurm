#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=gpu
#SBATCH --job-name=mt-14_wmt14ende37k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16g
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=2
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --array=0

trap_handler () {
    echo "Caught signal: " (
       # SIGTERM must be bypassed
          if [ "(" = "TERM" ]; then
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

module load cuda/10.1.243
source activate py37

split=$SLURM_ARRAY_TASK_ID
seeds=(1 101 65537 131071 524287 6700417)
seed=${seeds[$split]}

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

DATE=`date +%Y%m%d`
SAVE_ROOT=saved_models/transformer
DATA=/project/jonmay_231/max/data/wmt/wmt14_ende_vocab37k/data-bin
model=transformer_wmt_en_de
exp_name=14_apollo_wmt14ende37k_run${split}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}
src=en
tgt=de
cp $0 ${SAVE}/run.sh

python -u train.py ${DATA} \
    --seed ${seed} \
        --valid-subset valid \
            --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                --eval-tokenized-bleu --eval-bleu-detok "space" \
                    -a ${model} --optimizer apollo --lr 10 --clip-norm 0.1 --clip-mode 'each' -s $src -t $tgt \
                        --label-smoothing 0.1 --max-tokens 8192 --max-sentences 1024 --share-all-embeddings \
                            --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
                                --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 300000 450000 \
                                    --weight-decay 1e-8 --weight-decay-type 'decoupled' \
                                        --criterion label_smoothed_cross_entropy --max-update 500000 \
                                            --warmup-updates 1000 --warmup-init-lr 0.01 \
                                                --apollo-beta 0.9 --apollo-eps 1e-4 --save-dir ${SAVE} \
                                                    --keep-last-epochs 5 --keep-interval-updates 1 --update-freq 4 --save-interval-updates 5000 \
                                                        --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE}/log.txt
          ")" ]
            )
}
