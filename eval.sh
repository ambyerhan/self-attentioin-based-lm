
PROBLEM=lmptb_10k
MODEL=attention_lm
HPARAMS=attention_lm_base


ROOT=/home/ambyer/SELF-LM/v1
DATA_DIR=$ROOT/databin
TRAIN_DIR=$ROOT/traindir
TRAIN_STEPS=1000
EVAL_STEPS=3
GPU_NUM=1
GPU_ID=0
KEEP_MAX=10

echo "problem="$PROBLEM
echo "model="$MODEL
echo "hparams="$HPARAMS
echo "train_steps="$TRAIN_STEPS
echo "data_dir="$DATA_DIR
echo "train_dir="$TRAIN_DIR
echo "eval_steps="$EVAL_STEPS


CUDA_VISIBLE_DEVICES=${GPU_ID} t2t-trainer \
	--data_dir=$DATA_DIR \
	--problems=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
	--output_dir=$TRAIN_DIR \
	--keep_checkpoint_max=$KEEP_MAX \
	--eval_steps=$EVAL_STEPS
