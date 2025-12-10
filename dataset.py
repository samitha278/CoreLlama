import torch 
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer

import os






class TextDataset(Dataset):

    def __init__(self,dataset,tokenizer,max_len):
        
        self.dataset = dataset

        self.tokenizer = tokenizer
        self.max_len = max_len 
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):

        text = self.dataset[idx]['text']

        # tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        l = len(tokens)

        # truncate if tokens long
        if self.max_len < l:
            tokens = tokens[:self.max_len]
        else: # pad (if max_len > l)
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_len-l)

        # convert to tensor
        tokens = torch.tensor(tokens)
        x = tokens[:-1]
        y = tokens[1:]
        
        return x,y
    






def get_dataloader(split,batch_size,max_len):

    local_path = "./TinyStories"
    dataset_path = "roneneldan/TinyStories"
    
    # Check if already downloaded
    if os.path.exists(local_path):
        print("Loading from disk...")
        dataset = load_from_disk(local_path)
    else:
        print("Downloading for the first time...")
        dataset = load_dataset(dataset_path)
        dataset.save_to_disk(local_path)
        print(f"Saved to {local_path}")
        
        
        
    data_split = dataset[split]   # train / validation
    
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    dataset_wrapper = TextDataset(data_split,tokenizer,max_len)
    
    data_loader = DataLoader(dataset_wrapper,batch_size=batch_size,shuffle=True)
    
    return data_loader
    
    
    
   
   
   
   
   
   
   
    
    
    
if __name__=='__main__':
    
    # local_path = "./TinyStories"
    # dataset_path = "roneneldan/TinyStories"
    
    batch_size = 4
    max_len = 512 
    
    print(next(iter(get_dataloader('train',batch_size,max_len))))