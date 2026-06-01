import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json

from utils import parse_global_args, parse_train_args, set_seed, get_local_time, load_data_and_vocab
from utils import load_train_dataset, load_valid_dataset
from data import Train_Dataset, Valid_Dataset 
from model import TDPM

def validate_ddp(model, val_loader, device, args, sid_map, trie, mask_token_id, reverse_sid_map):
    model.eval()
    local_metrics = {
        "HR@10": 0, "HR@20": 0,
        "NDCG@10": 0, "NDCG@20": 0,
    }
    local_samples = 0
    sid_len = args.sid_len

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", disable=(dist.get_rank() != 0), dynamic_ncols=True)
        for batch_seq, batch_pad_mask, _ in pbar:
            batch_seq, batch_pad_mask = batch_seq.to(device), batch_pad_mask.to(device)
            B, L = batch_seq.shape

            time_indices = torch.arange(1, (L // sid_len) + 1, device=device).repeat_interleave(sid_len)
            time_indices = time_indices.unsqueeze(0).expand(B, -1).float() 

            mask_float = (~batch_pad_mask).float()
            mean = (time_indices * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-8)
            diff_sq = ((time_indices - mean) ** 2) * mask_float
            std = torch.sqrt(diff_sq.sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-8) + 1e-8)
            norm_time_batch = (time_indices - mean) / (std + 1e-8)
            
            for i in range(B):
                seq, pad_mask = batch_seq[i], batch_pad_mask[i]
                norm_time = norm_time_batch[i].unsqueeze(0)
                valid_len = (~pad_mask).sum().item()
                if valid_len < sid_len: continue

                target_sids = tuple(seq[valid_len - sid_len : valid_len].tolist())
                gt_item_id = reverse_sid_map.get(target_sids, None)
                if gt_item_id is None: continue 
                
                masked_seq = seq.clone()
                mask_indices = list(range(valid_len - sid_len, valid_len))
                for idx in mask_indices: masked_seq[idx] = mask_token_id
                
                candidates = [(masked_seq, 0.0, mask_indices.copy(), trie)]

                for step_idx in range(sid_len):
                    num_cands = len(candidates)
                    if num_cands == 0: break
                    cand_seqs = torch.stack([c[0] for c in candidates])
                    cand_pad_masks = pad_mask.unsqueeze(0).expand(num_cands, -1)
                    cand_time_embed = norm_time.expand(num_cands, -1)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        batch_logits = model.module(cand_seqs, cand_pad_masks, time_embed=cand_time_embed)

                    new_candidates = []
                    for cand_idx, (cand_seq, cand_score, rem_indices, cand_trie_node) in enumerate(candidates):
                        logits = batch_logits[cand_idx].float()
                        target_idx = rem_indices[0]
                        valid_vocab = list(cand_trie_node.keys())
                        target_logits = logits[target_idx].clone()
                        invalid_mask = torch.ones_like(target_logits, dtype=torch.bool)
                        invalid_mask[valid_vocab] = False
                        target_logits[invalid_mask] = -float('inf')

                        k = min(args.val_beam_size, len(valid_vocab))
                        if k == 0: continue
                        top_log_probs, top_tokens = F.log_softmax(target_logits, dim=-1).topk(k)

                        for log_p, tok in zip(top_log_probs, top_tokens):
                            new_seq = cand_seq.clone()
                            new_seq[target_idx] = tok.item()
                            new_candidates.append((new_seq, cand_score + log_p.item(), rem_indices[1:], cand_trie_node[tok.item()]))
                    candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:args.val_beam_size]

                item_score_dict = {}
                for cand_seq, score, _, _ in candidates:
                    gen_sids = tuple(cand_seq[valid_len - sid_len : valid_len].tolist())
                    item_id = reverse_sid_map.get(gen_sids, None)
                    if item_id is not None:
                        item_score_dict[item_id] = max(item_score_dict.get(item_id, -float('inf')), score)

                sorted_items = sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)
                top_k = args.topk
                predicted_items = [x[0] for x in sorted_items[:top_k]]
                
                local_samples += 1
                if gt_item_id in predicted_items:
                    rank = predicted_items.index(gt_item_id) + 1
                    if 1 <= rank <= 10: 
                        local_metrics["HR@10"] += 1
                        local_metrics["NDCG@10"] += 1.0 / np.log2(rank + 1)

                    if 1 <= rank <= 20: 
                        local_metrics["HR@20"] += 1
                        local_metrics["NDCG@20"] += 1.0 / np.log2(rank + 1)

    res = torch.tensor([local_metrics["HR@10"], local_metrics["HR@20"],
                        local_metrics["NDCG@10"], local_metrics["NDCG@20"], local_samples], 
                       dtype=torch.float32, device=device)

    dist.all_reduce(res, op=dist.ReduceOp.SUM)
    total_s = res[4].item() if res[4].item() > 0 else 1
    return {
        "HR@10": res[0].item()/total_s, "HR@20": res[1].item()/total_s,
        "NDCG@10": res[2].item()/total_s, "NDCG@20": res[3].item()/total_s
    }


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if global_rank == 0: 
        print(f"[DEBUG] {get_local_time()} DDP Initialized.")

    parser = argparse.ArgumentParser(description="Training args")
    parse_global_args(parser)
    parse_train_args(parser)
    args = parser.parse_args()
    
    lambda_k = args.lambda_k
    lambda_start = args.lambda_start
    warm_up_epochs = args.warm_up_epochs
    semantic_deviation_path = args.semantic_deviation_path
    
    if global_rank == 0: 
        print(args)

    set_seed(args.seed)

    interactions, sid_map, total_vocab_size, mask_token_id, pad_token_id = load_data_and_vocab(
        args.inter_path, args.sid_path, args.sid_len
    )
    reverse_sid_map = {tuple(v): k for k, v in sid_map.items()}
    
    if global_rank == 0: 
        print(f"[DEBUG] {get_local_time()} Loading Semantic Deviation from {semantic_deviation_path}...")
    with open(semantic_deviation_path, 'r') as f:
        user_semantic_deviation = json.load(f)

    if global_rank == 0: 
        print(f"[DEBUG] {get_local_time()} Building Prefix Tree Trie...")

    trie = {}

    for sids in sid_map.values():
        node = trie
        for s in sids:
            if s not in node: node[s] = {}
            node = node[s]

    train_dataset = Train_Dataset(interactions, sid_map, args.max_seq_len, args.sid_len, pad_token_id)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    val_dataset = Valid_Dataset(interactions, sid_map, args.max_seq_len, args.sid_len, pad_token_id)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    model = TDPM(args, total_vocab_size).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), 
        epochs=args.epochs, pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    best_res = 0.0
    sid_len = args.sid_len

    if global_rank == 0: 
        print(f"[DEBUG] {get_local_time()} Training Start!")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        
        is_warmup = epoch < warm_up_epochs
        curr_lambda = 1.0
        if not is_warmup:
            progress = (epoch - warm_up_epochs) / (args.epochs - warm_up_epochs)
            curr_lambda = lambda_start + (1.0 - lambda_start) * (progress ** lambda_k)
        
        stage_name = "Warm-up" if is_warmup else f"Time-Aware(λ={curr_lambda:.2f})"
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}[{stage_name}]", disable=(global_rank != 0), dynamic_ncols=True)
        
        for step, (batch_seq, batch_pad_mask, batch_uids) in enumerate(pbar):
            batch_seq, batch_pad_mask = batch_seq.to(device), batch_pad_mask.to(device)
            B, L = batch_seq.shape
            optimizer.zero_grad()
            
            mask_float = (~batch_pad_mask).float()
            last_token_indices = mask_float.sum(dim=1).long()

            if is_warmup:
                t = torch.empty(B, 1, device=device).uniform_(0.0, 0.5)
                mask_indices = (torch.rand(B, L, device=device) < t) & (~batch_pad_mask)
                
                input_seq = batch_seq.clone()
                input_seq[mask_indices] = mask_token_id
                labels = batch_seq.clone()
                labels[~mask_indices] = -100 
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(input_seq, src_key_padding_mask=batch_pad_mask)
                    raw_loss = criterion(logits.view(-1, total_vocab_size), labels.view(-1))
                    loss = raw_loss.sum() / (mask_indices.sum() + 1e-8)
                curr_avg_prob = t.mean().item()

            else:
                time_indices = torch.arange(1, (L // sid_len) + 1, device=device).repeat_interleave(sid_len)
                time_indices = time_indices.unsqueeze(0).expand(B, -1).float() 
                mean = (time_indices * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-8)
                std = torch.sqrt(torch.abs((time_indices - mean)**2 * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-8) + 1e-8)
                norm_time = (time_indices - mean) / (std + 1e-8)
                period_preference = torch.sigmoid(norm_time) 

                point_preference = torch.zeros((B, L), device=device)
                for b in range(B):
                    uid_str = str(batch_uids[b][0].item())
                    semantic_deviation = np.array(user_semantic_deviation.get(uid_str, [0.0] * (L // sid_len)))
                    
                    d_min, d_max = semantic_deviation.min(), semantic_deviation.max()
                    if d_max > d_min:
                        semantic_deviation = (semantic_deviation - d_min) / (d_max - d_min)
                    
                    token_deviation = torch.tensor(semantic_deviation, device=device).repeat_interleave(sid_len)
                    d_len = min(token_deviation.size(0), L)
                    point_preference[b, :d_len] = token_deviation[:d_len]

                pi = args.alpha + args.beta * (curr_lambda * period_preference + (1.0 - curr_lambda) * point_preference)

                mask_indices = (torch.rand(B, L, device=device) < pi) & (~batch_pad_mask)
                # full mask of the last SID
                for b in range(B):
                    idx_end = last_token_indices[b]
                    if idx_end >= sid_len: mask_indices[b, idx_end - sid_len : idx_end] = True
                
                input_seq = batch_seq.clone()
                input_seq[mask_indices] = mask_token_id
                labels = batch_seq.clone()
                labels[~mask_indices] = -100 
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(input_seq, src_key_padding_mask=batch_pad_mask, time_embed=norm_time)
                    raw_loss = criterion(logits.view(-1, total_vocab_size), labels.view(-1)).view(B, L)
                    
                    avg_pi = pi[mask_indices].mean()
                    normalized_pi = pi / (avg_pi + 1e-8)
                    loss = (raw_loss * normalized_pi).sum() / (mask_indices.sum() + 1e-8)
                curr_avg_prob = pi[~batch_pad_mask].mean().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            curr_loss = loss.item()
            total_loss += curr_loss
            if global_rank == 0:
                pbar.set_postfix({"loss": f"{curr_loss:.3f}", "p_avg": f"{curr_avg_prob:.2f}"})

        if global_rank == 0:
            desc = "Warm_up" if is_warmup else "Time-Aware"
            tqdm.write(f"[{get_local_time()}] Epoch {epoch+1} ({desc}) | Train Loss: {total_loss/len(train_loader):.4f}")

        if (not is_warmup and (epoch + 1) % args.valid_interval == 0) or epoch == args.epochs - 1:
            if global_rank == 0: 
                tqdm.write(f"\n[{get_local_time()}] Epoch {epoch+1} done. Validating...")

            val_res = validate_ddp(model, val_loader, device, args, sid_map, trie, mask_token_id, reverse_sid_map)
            cur_res = val_res['HR@10'] + val_res['HR@20'] + val_res['NDCG@10'] + val_res['NDCG@20']

            if global_rank == 0:
                avg_train_loss = total_loss / len(train_loader)
                tqdm.write(f"[Validation] {get_local_time()} Result:")
                tqdm.write(f"HR@10: {val_res['HR@10']:.4f} | HR@20: {val_res['HR@20']:.4f}")
                tqdm.write(f"NDCG@10: {val_res['NDCG@10']:.4f} | NDCG@20: {val_res['NDCG@20']:.4f}")

                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save(model.module.state_dict(), save_path)
                
                if cur_res > best_res:
                    best_res = cur_res
                    torch.save(model.module.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
                    tqdm.write(f"New Best Model Saved!")
                tqdm.write("-" * 50)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()