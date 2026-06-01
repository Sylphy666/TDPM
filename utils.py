import os
import time
import random
import argparse
import torch
import numpy as np
import json
from data import Train_Dataset, Valid_Dataset, Test_Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=20, help="Max items in sequence")
    parser.add_argument("--inter_path", type=str, required=True, help="Path to interactions JSON")

    parser.add_argument("--sid_path", type=str, required=True, help="Path to SID JSON")
    parser.add_argument("--sid_len", type=int, default=4, help="Number of SID tokens per item")

    parser.add_argument("--topk", type=int, default=20)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.25)


def parse_train_args(parser):
    parser.add_argument("--semantic_deviation_path", type=str, default=None, help="Path to semantic_deviation.json")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--valid_interval", type=int, default=40)
    parser.add_argument("--val_beam_size", type=int, default=50)

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=120, help="Total number of training epochs")
    parser.add_argument("--warm_up_epochs", type=int, default=20, help="Epochs of warm-up stage")
    parser.add_argument("--alpha", type=float, default=0.1, help="Base masking probability")
    parser.add_argument("--beta", type=float, default=0.3, help="Scaling factor for fused preference")
    parser.add_argument("--lambda_start", type=float, default=0.6, help="Initial value of lambda")
    parser.add_argument("--lambda_k", type=float, default=2.0, help="Curvature of growth schedule of lambda")


def parse_test_args(parser):
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)


def load_data_and_vocab(inter_path, sid_path, sid_len):
    with open(inter_path, 'r') as f:
        interactions = json.load(f)
    with open(sid_path, 'r') as f:
        raw_sid_map = json.load(f)
        
    vocab_offsets = [0] * sid_len
    max_vals = [0] * sid_len
    
    for sids in raw_sid_map.values():
        for i in range(sid_len):
            max_vals[i] = max(max_vals[i], int(sids[i]))
            
    current_offset = 0
    for i in range(sid_len):
        vocab_offsets[i] = current_offset
        current_offset += (max_vals[i] + 1)
        
    sid_map = {}
    for item_id, sids in raw_sid_map.items():
        sid_map[str(item_id)] = [int(sids[i]) + vocab_offsets[i] for i in range(sid_len)]
        
    vocab_size = current_offset
    mask_token_id = vocab_size
    pad_token_id = vocab_size + 1
    total_vocab_size = pad_token_id + 1
    
    return interactions, sid_map, total_vocab_size, mask_token_id, pad_token_id

def load_train_dataset(interactions, sid_map, args, mask_token_id, pad_token_id):
    return Train_Dataset(interactions, sid_map, args.max_seq_len, args.sid_len, pad_token_id)

def load_valid_dataset(interactions, sid_map, args, mask_token_id, pad_token_id):
    return Valid_Dataset(interactions, sid_map, args.max_seq_len, args.sid_len, pad_token_id)

def load_test_dataset(interactions, sid_map, args, mask_token_id, pad_token_id):
    return Test_Dataset(interactions, sid_map, args.max_seq_len, args.sid_len, pad_token_id)