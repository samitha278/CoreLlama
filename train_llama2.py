import torch
import torch.nn as nn

from dataset import get_dataloader
from modeling_llama2 import LLaMA2Config,LlamaForCausalLM


device = 'cuda' if torch.cuda.is_available() else 'cpu'



batch_size = 4
max_len = 512

# data loaders ---------------------------------------------------
train_loader = get_dataloader('train',batch_size,max_len)
val_loader = get_dataloader('validation',batch_size,max_len)




# create model ---------------------------------------------------
config = LLaMA2Config()
config.max_seq_len = max_len

llama = LlamaForCausalLM(config).to(device)

# n_param = sum([p.nelement() for p in llama.parameters()])
# print(n_param)    - 1574241536 ~ 1.5B params

import sys;exit(0)




# train loop  
    # validation
    # save checkpoints
    # save logs - losses

max_iter = 10000


for i in range(max_iter):
    
    pass
    
    
 