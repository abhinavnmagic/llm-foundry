

#MODEL_NAME_OR_PATH="/home/abhinav/llm-foundry/scripts/train/output_dir/dense_finetuning_mosaicml/mpt-7b-instruct_2ep_batch_32_LR_1e-5/hf/"
MAX_DURATION=2ep
GLOBAL_TRAIN_BATCH_SIZE=32
MICROBATCH_SIZE=2
LR=1e-5
#MODEL_NAME_OR_PATH="/home/abhinav/llm-foundry/scripts/train/output_dir/dense_finetuning_mosaicml/mpt-7b-instruct_2ep_batch_32_LR_1e-5/oneshot_0.5/hf/"
MODEL_NAME_OR_PATH="/home/abhinav/llm-foundry/scripts/train/output_dir/dense_finetuning_mosaicml/mpt-7b-instruct_2ep_batch_32_LR_1e-5/oneshot_0.4/hf/"
PRECISION="amp_bf16"
export RUN_NAME="sparse_finetuning_40sp_"${MODEL_NAME_OR_PATH}"_"${MAX_DURATION}"_batch_"${GLOBAL_TRAIN_BATCH_SIZE}"_LR_"${LR}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_ENTITY=neuralmagicml
export WANDB_PROJECT=mpt_platypus_sparse_ft
export WANDB_DISABLED=False

composer train_sparse_KD.py yamls/finetune/mpt/FT_platypus.yaml \
	model_name_or_path=${MODEL_NAME_OR_PATH} \
	load_path=${LOAD_PATH} \
	max_duration=${MAX_DURATION} \
	global_train_batch_size=${GLOBAL_TRAIN_BATCH_SIZE} \
	device_train_microbatch_size=${MICROBATCH_SIZE} \
	optimizer.lr=${LR} \
	precision=${PRECISION}
