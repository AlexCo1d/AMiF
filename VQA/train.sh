export CUDA_VISIBLE_DEVICES=2,3;
python -m torch.distributed.launch --nnodes=1 --master_port 2381 --nproc_per_node=2 train.py \
--num_workers 8 \
--batch_size 28 \
--epochs 60 \
--eval_freq 1 \
--vit_type vit_16 \
--warmup_epochs 0 \
--img_size 384 \
--classifier_vqa \
--distill_model \
--eval_batch_size 24 \
--lr 2e-5 \
--min_lr 1e-6 \
--dataset_use radvqa --dataset_path /home/data/Jingkai/alex/radvqa \
--checkpoint /home/data/Jingkai/alex/weight/MM-all-40-x.pth --start_epoch 0 \
--output_dir /home/data/Jingkai/alex/vqa_out_radx