export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=23322 test_ddp.py \
    --beam_size 50 \
    --log_interval 10 \
    --batch_size 128 \
    --sid_len 4 \
    --inter_path ./data/Beauty/Beauty.inter.json \
    --sid_path ./data/Beauty/Beauty.sid.json \
    --ckpt_path ./ckpt/Beauty/best_model.pth \
