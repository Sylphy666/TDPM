import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

CONFIG = {
    "emb_path": "../data/Beauty/Beauty.item.embedding.pt",
    "inter_path": "../data/Beauty/Beauty.inter.json",
    "output_path": "../data/Beauty/Beauty.semantic_deviation.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def generate_deviation_map():
    print(f"Loading item embeddings from {CONFIG['emb_path']}...")
    if not os.path.exists(CONFIG['emb_path']):
        print("Error: Embedding file not found!")
        return
    item_embs = torch.load(CONFIG['emb_path'], map_location=CONFIG['device'])
    for k in item_embs:
        item_embs[k] = item_embs[k].float()
    
    print(f"Loading interactions from {CONFIG['inter_path']}...")
    with open(CONFIG['inter_path'], 'r') as f:
        interactions = json.load(f)
    
    user_deviation_map_raw = {}
    all_deviation_values = []
    
    print("Calculating raw semantic deviations...")
    for uid, items in tqdm(interactions.items()):
        deviations = [0.0]
        
        for i in range(1, len(items)):
            item_curr = str(items[i])
            item_prev = str(items[i-1])
            
            if item_curr in item_embs and item_prev in item_embs:
                vec_curr = item_embs[item_curr].unsqueeze(0)
                vec_prev = item_embs[item_prev].unsqueeze(0)
                
                sim = F.cosine_similarity(vec_curr, vec_prev).item()
                deviation = 1.0 - sim
                deviations.append(float(deviation))
                all_deviation_values.append(deviation)
            else:
                deviations.append(0.0)
        
        user_deviation_map_raw[str(uid)] = deviations

    if not all_deviation_values:
        print("No deviation values calculated!")
        return

    min_deviation = min(all_deviation_values)
    max_deviation = max(all_deviation_values)
    print(f"\nRaw Statistics - Min Deviation: {min_deviation:.4f}, Max Deviation: {max_deviation:.4f}")
    
    final_deviation_map = {}
    print("Applying Min-Max Scaling to normalize deviation signals to [0, 1]...")
    
    for uid, deviations in user_deviation_map_raw.items():
        scaled_deviations = []
        for d in deviations:
            if d == 0:
                scaled_deviations.append(0.0)
            else:
                s = (d - min_deviation) / (max_deviation - min_deviation + 1e-8)
                scaled_deviations.append(round(float(s), 4))
        final_deviation_map[uid] = scaled_deviations

    print(f"Saving normalized deviation map to {CONFIG['output_path']}...")
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    with open(CONFIG['output_path'], 'w') as f:
        json.dump(final_deviation_map, f)
    
    all_scaled = [d for deviations in final_deviation_map.values() for d in deviations if d > 0]
    print(f"\nNormalized Statistics:")
    print(f"Total Users: {len(final_deviation_map)}")
    print(f"Avg Normalized Deviation: {np.mean(all_scaled):.4f}")
    print(f"Max Normalized Deviation: {max(all_scaled) if all_scaled else 0.0:.4f}")
    print(f"Min Normalized Deviation: {min(all_scaled) if all_scaled else 0.0:.4f}")

if __name__ == "__main__":
    generate_deviation_map()
