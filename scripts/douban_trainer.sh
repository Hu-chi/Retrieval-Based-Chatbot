dir=$(dirname "$PWD")
cd "$dir" || exit
echo "当前目录: $PWD"
export PYTHONPATH=${PYTHONPATH}:$dir/src:$dir
echo "$PYTHONPATH"
#python src/offline_training/douban_trainer.py \
#CUDA_VISIBLE_DEVICES=4 \
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 src/offline_training/douban_trainer.py \
#  --encoder bert \
#  --pretrain_checkpoint checkpoint/bert_base_chinese \
#  --data_dir data/DoubanConversaionCorpus \
#  --cache_dir data/cache/douban \
#  --train_batch_size 4 \
#  --valid_batch_size 4 \
#  --world_size 1

CUDA_VISIBLE_DEVICES=4 \
  python src/offline_training/douban_trainer.py \
  --encoder bert \
  --pretrain_checkpoint checkpoint/bert_base_chinese \
  --data_dir data/DoubanConversaionCorpus \
  --cache_dir data/cache/douban \
  --train_batch_size 4 \
  --valid_batch_size 4 \
  --world_size 1
