import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

CONFIG = {
    "model_path": "../../Qwen3-Embedding-8B", 
    "item_path": "../data/Beauty/Beauty.item.json",
    "output_path": "../data/Beauty/Beauty.item.embedding.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "max_text_len": 256,
}

def load_data():
    print(f"Loading items from {CONFIG['item_path']}...")
    with open(CONFIG['item_path'], 'r') as f:
        items_dict = json.load(f)
    return items_dict

def get_item_text(item):
    title = item.get("title", "")
    brand = item.get("brand", "")
    content = f"{brand} {title}" if brand else title
    
    instruction = "Represent this product for recommendation system clustering: "
    return instruction + content

@torch.no_grad()
def generate_embeddings():
    items_raw = load_data()
    item_ids = list(items_raw.keys())
    
    print(f"Loading Embedding Model from {CONFIG['model_path']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'], trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        CONFIG['model_path'], 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    item_texts = []
    for iid in item_ids:
        item_texts.append(get_item_text(items_raw[iid]))

    item_emb_map = {}

    print(f"Generating item embeddings for {len(item_ids)} items...")
    for i in tqdm(range(0, len(item_texts), CONFIG['batch_size'])):
        batch_ids = item_ids[i : i + CONFIG['batch_size']]
        batch_texts = item_texts[i : i + CONFIG['batch_size']]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=CONFIG['max_text_len'], 
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_embs = last_hidden_state[torch.arange(last_hidden_state.size(0)), sequence_lengths]
        
        batch_embs = F.normalize(batch_embs, p=2, dim=1)
        
        batch_embs = batch_embs.cpu().float()
        
        for iid, emb in zip(batch_ids, batch_embs):
            item_emb_map[str(iid)] = emb

    print(f"Saving item embeddings to {CONFIG['output_path']}...")
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    torch.save(item_emb_map, CONFIG['output_path'])
    print(f"Success! Total items embedded: {len(item_emb_map)}")

if __name__ == "__main__":
    generate_embeddings()
