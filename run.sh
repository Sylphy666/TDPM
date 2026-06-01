export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=23325 main.py \
    --seed 42 \
    --lr 3e-3 \
    --dropout 0.25 \
    --weight_decay 1e-3 \
    --batch_size 128 \
    --inter_path ./data/Beauty/Beauty.inter.json \
    --output_dir ./ckpt/Beauty \
    --sid_path ./data/Beauty/Beauty.sid.json \
    --semantic_deviation_path ./data/Beauty/Beauty.semantic_deviation.json \
    --epochs 120 \
    --warm_up_epoch 60 \
    --valid_interval 20 \
    --val_beam_size 50 \
    --alpha 0.1 \
    --beta 0.3 \
    --lambda_start 0.5 \
    --lambda_k 2.0 \
