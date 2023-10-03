

#MODEL_NAME_OR_PATH="mosaicml/mpt-7b-instruct"
MODEL_NAME_OR_PATH="mosaicml/mpt-7b"
MAX_DURATION=2ep
GLOBAL_TRAIN_BATCH_SIZE=32
#LR=1e-5
#export RUN_NAME="dense_finetuning_base_"${MODEL_NAME_OR_PATH}"_"${MAX_DURATION}"_batch_"${GLOBAL_TRAIN_BATCH_SIZE}"_LR_"${LR}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_ENTITY=neuralmagicml
export WANDB_PROJECT=mpt_platypus_dense_ft
export WANDB_DISABLED=False
MICROBATCH_SIZE=4
PRECISION="amp_bf16"

for GRAD_CLIP in 3.0 5.0;
do
    for LR in 1e-5;
    do
	#export GRAD_CLIP=2.0
        export RUN_NAME="dense_finetuning_base_"${MODEL_NAME_OR_PATH}"_"${MAX_DURATION}"_batch_"${GLOBAL_TRAIN_BATCH_SIZE}"_LR_"${LR}"_clip_"${GRAD_CLIP}
        composer train.py yamls/finetune/mpt/FT_platypus.yaml \
	    model_name_or_path=${MODEL_NAME_OR_PATH} \
	    max_duration=${MAX_DURATION} \
	    global_train_batch_size=${GLOBAL_TRAIN_BATCH_SIZE} \
	    microbatch_size=${MICROBATCH_SIZE} \
	    precision=${PRECISION} \
	    optimizer.lr=${LR} \
	    algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP}
    done
done
exit

