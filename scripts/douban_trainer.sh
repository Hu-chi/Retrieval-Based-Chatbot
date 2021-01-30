dir=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$dir" || exit
echo "当前目录: $PWD"
export PYTHONPATH="$PYTHONPATH:$dir/src:$dir"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501

CUDA_VISIBLE_DEVICES=1 \
  python \
  src/offline_training/douban_trainer.py \
  --encoder bert \
  --pretrain_checkpoint checkpoint/bert_base_chinese \
  --data_dir data/DoubanConversaionCorpus \
  --cache_dir data/cache/douban \
  --model_checkpoint checkpoint/douban_pair_checkpoint \
  --gradient_accumulation_steps 2 \
  --train_batch_size 3 \
  --valid_batch_size 3 \
  --world_size 3 \
  --num_worker 2 \
  --seed 42
#  --eval_before_start \
#  --best_acc 0.81504 \
