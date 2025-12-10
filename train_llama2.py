import torch
import torch.nn as nn

from dataset import get_dataloader
from modeling_llama2 import LLaMA2Config,LlamaForCausalLM


device = 'cuda' if torch.cuda.is_available() else 'cpu'



max_iter = 10
batch_size = 4
max_len = 512

# data loaders ---------------------------------------------------
train_loader = get_dataloader('train',batch_size,max_len)
# val_loader = get_dataloader('validation',batch_size,max_len)




# create model ---------------------------------------------------
config = LLaMA2Config()
config.max_seq_len = max_len

llama = LlamaForCausalLM(config).to(device)

n_param = sum([p.nelement() for p in llama.parameters()])
print(f"total parameters: {n_param/10**9}B")    # - 1574241536 ~ 1.5B params




# train loop ------------------------------------------------------
    # loss,backward,update - done
    # validation
    # save checkpoints
    # save logs - losses

optimizer = torch.optim.AdamW(llama.parameters())
cross_el = nn.CrossEntropyLoss()


for i in range(max_iter):
    
    x,y = next(iter(train_loader))
    x,y = x.to(device),y.to(device)
    
    logits = llama(x,0)
    loss = cross_el(logits.view(-1,config.vocab_size),y.view(-1))  # flatten
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss.item())
