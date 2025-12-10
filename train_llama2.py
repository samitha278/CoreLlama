import torch
import torch.nn as nn


from modeling_llama2 import LLaMA2Config,LlamaForCausalLM



device = 'cuda' if torch.cuda.is_available() else 'cpu'




# data loaders -----------------------------------------







# create model
config = LLaMA2Config()
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
    
    
 