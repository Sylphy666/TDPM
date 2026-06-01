import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

from utils import parse_global_args, parse_test_args, set_seed, get_local_time, load_data_and_vocab, load_test_dataset
from model import TDPM

def test_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    parser = argparse.ArgumentParser(description="Testing args")
    parse_global_args(parser)
    parse_test_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)

    if global_rank == 0:
        print(args)
        print(f"{get_local_time()} === Evaluation ===")

    interactions, sid_map, total_vocab_size, mask_token_id, pad_token_id = load_data_and_vocab(
        args.inter_path, args.sid_path, args.sid_len
    )
    
    reverse_sid_map = {tuple(sids): item_id for item_id, sids in sid_map.items()}
    
    trie = {}
    level_valid_vocabs =[set() for _ in range(args.sid_len)]
    for sids in sid_map.values():
        node = trie
        for i, sid in enumerate(sids):
            level_valid_vocabs[i].add(sid)
            if sid not in node:
                node[sid] = {}
            node = node[sid]

    test_dataset = load_test_dataset(interactions, sid_map, args, mask_token_id, pad_token_id)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4)

    model = TDPM(args, total_vocab_size).to(device)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    metrics = {
        "HR@10": 0, "HR@20": 0,
        "NDCG@10": 0, "NDCG@20": 0
    }
    total_samples = 0
    total_valid_candidates = 0
    total_generated_candidates = 0

    pbar = tqdm(test_loader, disable=(global_rank != 0), desc="Testing")
    
    with torch.no_grad():
        for step, (batch_seq, batch_pad_mask, _) in enumerate(pbar):
            batch_seq = batch_seq.to(device)
            batch_pad_mask = batch_pad_mask.to(device)
            B = batch_seq.size(0)
            
            L = batch_seq.size(1)
            sid_len = args.sid_len

            time_indices = torch.arange(1, (L // sid_len) + 1, device=device).repeat_interleave(sid_len)
            time_indices = time_indices.unsqueeze(0).float()

            mask_float = (~batch_pad_mask).float()
            mean = (time_indices * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-8)
            diff_sq = ((time_indices - mean) ** 2) * mask_float

            std = torch.sqrt(diff_sq.sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-8) + 1e-8)
            norm_time_batch = (time_indices - mean) / (std + 1e-8) # (B, L)
            
            for i in range(B):
                seq = batch_seq[i]
                pad_mask = batch_pad_mask[i]
                norm_time = norm_time_batch[i].unsqueeze(0) # (1, L)
                
                valid_len = (~pad_mask).sum().item()
                if valid_len < args.sid_len: continue
                    
                target_sids = seq[valid_len - args.sid_len : valid_len]
                gt_item_id = reverse_sid_map.get(tuple(target_sids.tolist()), None)
                if gt_item_id is None: 
                    continue 
                
                masked_seq = seq.clone()
                mask_indices = list(range(valid_len - args.sid_len, valid_len))
                for idx in mask_indices:
                    masked_seq[idx] = mask_token_id
                
                candidates =[(masked_seq, 0.0, mask_indices.copy(), trie)]
                
                for step_idx in range(args.sid_len):
                    num_cands = len(candidates)
                    if num_cands == 0: break
                    
                    cand_seqs = torch.stack([c[0] for c in candidates])
                    cand_pad_masks = pad_mask.unsqueeze(0).expand(num_cands, -1)
                    cand_time_embed = norm_time.expand(num_cands, -1)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        batch_logits = model.module(cand_seqs, cand_pad_masks, time_embed=cand_time_embed)
                    
                    new_candidates =[]
                    for cand_idx, (cand_seq, cand_score, rem_indices, cand_trie_node) in enumerate(candidates):
                        logits = batch_logits[cand_idx].float()
                        
                        target_idx = rem_indices[0]
                        valid_vocab = list(cand_trie_node.keys())
                        
                        target_logits = logits[target_idx].clone()
                        invalid_mask = torch.ones_like(target_logits, dtype=torch.bool)
                        invalid_mask[valid_vocab] = False
                        target_logits[invalid_mask] = -float('inf')
                        
                        k_to_extract = min(args.beam_size, len(valid_vocab))
                        if k_to_extract == 0: continue
                            
                        top_log_probs, top_tokens = F.log_softmax(target_logits, dim=-1).topk(k_to_extract)
                        
                        for log_p, tok in zip(top_log_probs, top_tokens):
                            if log_p == -float('inf'): 
                                continue

                            tok_val = tok.item()
                            
                            new_seq = cand_seq.clone()
                            new_seq[target_idx] = tok
                            new_rem = rem_indices.copy()
                            new_rem.remove(target_idx)
                            
                            new_node = cand_trie_node.get(tok_val, {})
                            new_candidates.append((new_seq, cand_score + log_p.item(), new_rem, new_node))
                            
                    candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:args.beam_size]
                
                item_score_dict = {}
                for cand_seq, score, _, _ in candidates:
                    total_generated_candidates += 1
                    gen_sids = tuple(cand_seq[valid_len - args.sid_len : valid_len].tolist())
                    item_id = reverse_sid_map.get(gen_sids, None)
                    
                    if item_id is not None:
                        total_valid_candidates += 1
                        if item_id not in item_score_dict:
                            item_score_dict[item_id] = score
                        else:
                            item_score_dict[item_id] = max(item_score_dict[item_id], score)

                sorted_items_list = sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)
                predicted_items = [x[0] for x in sorted_items_list[:args.topk]]

                total_samples += 1
                
                rank = predicted_items.index(gt_item_id) + 1 if gt_item_id in predicted_items else -1
                    
                if 1 <= rank <= 10: 
                    metrics["HR@10"] += 1
                    metrics["NDCG@10"] += 1.0 / np.log2(rank + 1)

                if 1 <= rank <= 20: 
                    metrics["HR@20"] += 1
                    metrics["NDCG@20"] += 1.0 / np.log2(rank + 1)

            if global_rank == 0 and total_samples > 0:
                cur_hr10 = metrics["HR@10"] / total_samples
                cur_hr20 = metrics["HR@20"] / total_samples

                cur_ndcg10 = metrics["NDCG@10"] / total_samples
                cur_ndcg20 = metrics["NDCG@20"] / total_samples
                
                pbar.set_postfix({"HR@10": f"{cur_hr10:.4f}", "HR@20": f"{cur_hr20:.4f}", "NDCG@10": f"{cur_ndcg10:.4f}", "NDCG@20": f"{cur_ndcg20:.4f}"})

    local_metrics = torch.tensor([
        metrics["HR@10"], metrics["HR@20"],
        metrics["NDCG@10"], metrics["NDCG@20"], total_samples
    ], dtype=torch.float32, device=device)
    
    dist.reduce(local_metrics, dst=0, op=dist.ReduceOp.SUM)
    
    if global_rank == 0:
        samps = local_metrics[4].item()
        if samps > 0:
            print("="*45)
            print(f"Overall Performance")
            print(f"HR@10     : {local_metrics[0].item()/samps:.4f}")
            print(f"HR@20     : {local_metrics[1].item()/samps:.4f}")
            print(f"NDCG@10   : {local_metrics[2].item()/samps:.4f}")
            print(f"NDCG@20   : {local_metrics[3].item()/samps:.4f}")
            print("="*45)

    dist.destroy_process_group()

if __name__ == "__main__":
    test_ddp()