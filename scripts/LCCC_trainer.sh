dir=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$dir" || exit
echo "当前目录: $PWD"
export PYTHONPATH=${PYTHONPATH}:$dir/src:$dir

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501

CUDA_VISIBLE_DEVICES=1 \
  python src/offline_training/LCCC_trainer.py \
  --encoder bert \
  --pretrain_checkpoint checkpoint/bert_base_chinese \
  --data_dir data/LCCC-base-split \
  --cache_dir data/cache/LCCC_base \
  --model_checkpoint checkpoint/LCCC_base_pair_checkpoint \
  --gradient_accumulation_steps 3 \
  --train_batch_size 3 \
  --valid_batch_size 3 \
  --world_size 1 \
  --num_worker 2 \
  --seed 42 \
#  --best_acc 0.84706
#  --eval_before_start \
