import torch
from torch.utils.data import Dataset

class Base_Dataset(Dataset):
    def __init__(self, interactions, sid_map, max_items, num_hierarchies, pad_token_id):
        self.sid_map = sid_map
        self.max_items = max_items
        self.num_hierarchies = num_hierarchies
        self.max_tokens = max_items * num_hierarchies
        self.pad_token_id = pad_token_id
        
        self.data = []
        self.uid_data = []

    def __len__(self):
        return len(self.data)
    
    def items_to_sids(self, items):
        sids = []
        for item in items:
            item_str = str(item)
            if item_str in self.sid_map:
                sids.extend(self.sid_map[item_str])
        return sids

    def pad_sequence(self, sid_seq):
        seq_len = len(sid_seq)
        if seq_len > self.max_tokens:
            sid_seq = sid_seq[-self.max_tokens:]
            seq_len = self.max_tokens
            
        padded_seq = sid_seq + [self.pad_token_id] * (self.max_tokens - seq_len)
        pad_mask = [False] * seq_len + [True] * (self.max_tokens - seq_len)
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(pad_mask, dtype=torch.bool)
    
    def __getitem__(self, idx):
        seq_tensor, mask_tensor = self.pad_sequence(self.data[idx])
        uid_val = self.uid_data[idx]

        try:
            uid_int = int(uid_val)
        except ValueError:

            uid_int = hash(uid_val) % (10**10) 
            
        return seq_tensor, mask_tensor, torch.tensor([uid_int], dtype=torch.long)

    
class Train_Dataset(Base_Dataset):
    def __init__(self, interactions, sid_map, max_items, num_hierarchies, pad_token_id):
        super().__init__(interactions, sid_map, max_items, num_hierarchies, pad_token_id)
        for uid, items in interactions.items():
            if len(items) >= 3:
                train_part = items[:-2] 
                for i in range(1, len(train_part)):
                    sub_sequence = train_part[:i+1]
                    if max_items > 0:
                        sub_sequence = sub_sequence[-max_items:]

                    sid_seq = self.items_to_sids(sub_sequence)
                    if len(sid_seq) > 0:
                        self.data.append(sid_seq)
                        self.uid_data.append(uid)


class Valid_Dataset(Base_Dataset):
    def __init__(self, interactions, sid_map, max_items, num_hierarchies, pad_token_id):
        super().__init__(interactions, sid_map, max_items, num_hierarchies, pad_token_id)
        for uid, items in interactions.items():
            if len(items) >= 3:
                valid_items = items[:-1]
                sid_seq = self.items_to_sids(valid_items)
                if len(sid_seq) > 0:
                    self.data.append(sid_seq)
                    self.uid_data.append(uid)


class Test_Dataset(Base_Dataset):
    def __init__(self, interactions, sid_map, max_items, num_hierarchies, pad_token_id):
        super().__init__(interactions, sid_map, max_items, num_hierarchies, pad_token_id)
        for uid, items in interactions.items():
            if len(items) >= 3:
                test_items = items[:]
                sid_seq = self.items_to_sids(test_items)
                if len(sid_seq) > 0:
                    self.data.append(sid_seq)
                    self.uid_data.append(uid)