dir=$(dirname "$PWD")
cd "$dir" || exit
echo "当前目录: $PWD"
export PYTHONPATH=${PYTHONPATH}:$dir/src:$dir

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501

CUDA_VISIBLE_DEVICES=0,1,2 \
  python -m torch.distributed.launch --nproc_per_node=3 \
  --master_port 29501 \
  src/offline_training/douban_trainer.py \
  --encoder bert \
  --pretrain_checkpoint checkpoint/bert_base_chinese \
  --data_dir data/DoubanConversaionCorpus \
  --cache_dir data/cache/douban \
  --model_checkpoint checkpoint/douban_pair_checkpoint \
  --gradient_accumulation_steps 1 \
  --train_batch_size 3 \
  --valid_batch_size 3 \
  --world_size 3 \
  --num_worker 2 \
  --seed 42
#  --best_acc 0.81504 \
#  --eval_before_start \
