import torch
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

CONFIG = {
    "emb_path": "../data/Beauty/Beauty.item.embedding.pt",
    "output_sid_path": "../data/Beauty_test/Beauty.sid.json",
    "sid_len": 4,
    "clusters_per_level": 256,
    "seed": 42
}

def generate_residual_kmeans_sids():
    print(f"Loading item embeddings from {CONFIG['emb_path']}...")
    item_emb_map = torch.load(CONFIG['emb_path'])
    
    item_ids = list(item_emb_map.keys())
    embeddings = np.stack([item_emb_map[iid].numpy() for iid in item_ids])
    
    N, D = embeddings.shape
    print(f"Total items: {N}, Embedding dimension: {D}")

    final_sids = {iid: [] for iid in item_ids}
    current_residuals = embeddings.copy()

    for level in range(CONFIG['sid_len']):
        print(f"Clustering Level {level+1} / {CONFIG['sid_len']}...")
        
        kmeans = KMeans(
            n_clusters=CONFIG['clusters_per_level'], 
            random_state=CONFIG['seed'], 
            n_init=10,
            max_iter=300,
            verbose=0
        )
        
        cluster_labels = kmeans.fit_predict(current_residuals)
        centroids = kmeans.cluster_centers_

        for i, iid in enumerate(item_ids):
            final_sids[iid].append(int(cluster_labels[i]))
        
        if level < CONFIG['sid_len'] - 1:
            assigned_centroids = centroids[cluster_labels]
            current_residuals = current_residuals - assigned_centroids
            
            res_norm = np.mean(np.linalg.norm(current_residuals, axis=1))
            print(f"   - Average residual norm after Level {level+1}: {res_norm:.4f}")

    print(f"Saving Item SID to {CONFIG['output_sid_path']}...")
    os.makedirs(os.path.dirname(CONFIG['output_sid_path']), exist_ok=True)
    with open(CONFIG['output_sid_path'], 'w') as f:
        json.dump(final_sids, f)
    
    print("\nSuccess! Sample SID format:")
    sample_id = item_ids[0]
    print(f"Item {sample_id} -> SIDs: {final_sids[sample_id]}")

if __name__ == "__main__":
    generate_residual_kmeans_sids()