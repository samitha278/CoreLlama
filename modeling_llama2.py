import torch
import torch.nn as nn



class LLaMA2Config():

    def __init__(self):

        self.n_embd = 4096
        self.n_layers = 32
        self.heads = 32           # for queries
        self.n_kv_heads = None    # for k v
        self.vocab_size = None
        self.n_hidden = None      # for mlp

        self.norm_eps = 1e-5

        self.max_batch_size = 32
        self.max_seq_len = 2048






class LlamaRMSNorm(nn.Module):

    def __init__(self,n_embd,norm_eps):
        super().__init__()
        self.norm_eps = norm_eps

        self.gamma = nn.Parameter(torch.ones(n_embd))

    def forward(self,x):

        rms = torch.sqrt(self.norm_eps + torch.mean(torch.pow(x,2.0),dim =-1,keepdim=True))

        x_norm = (x/rms) * self.gamma

        return x_norm







def precompute_theta_pos_frequencies(d_model,max_len,device,base=10000.0):

    indexes = torch.arange(0,d_model//2,    dtype= torch.float32)
    w_i = torch.pow(base, (-2*indexes)/d_model)

    m = torch.arange(0,max_len,    dtype= torch.float32)


    freq = torch.outer(m,w_i)  # [max_len,d_model//2]

    freq_complex = torch.polar(torch.ones_like(freq),freq)

    return freq_complex



def rope(freq_coomlex,x):

    B,T,C = x.shape
    x_complex = torch.view_as_complex(x.reshape(B,T,C//2,2)) # [x,y] -> x+iy

    x_rotated_complex = x_complex * freq_coomlex

    x_rotated_real = torch.view_as_real(x_rotated_complex)   # [B,T,C//2,2]

    return x_rotated_real.reshape(B,T,C)








class LlamaMLP(nn.Module):

    def __init__(self,config):
        super().__init__()

        n_hidden = int((4*config.n_embd) *(2/3)) # 2/3 to keep total params similar as org mlp
        n_hidden = 256*((n_hidden+255)//256)     # multiple of 256 for hardware efficiency

        self.W = nn.Linear(config.n_embd,n_hidden, bias=False)   # to gate
        self.silu = nn.SiLU()   # silu = x * sigmoid(x)

        self.V = nn.Linear(config.n_embd,n_hidden, bias=False)   # to value
        self.out = nn.Linear(n_hidden,config.n_embd, bias=False)


    def forward(self,x):

        gate = self.silu(self.W(x))
        value = self.V(x)

        out = self.out(gate * value)   # gate controls values : model learn it










class LlamaAttention(nn.Module):

    # grouped-query attention (GQA)

    # for training

    def __init__(self,config):
        super().__init__()

        self.config = config
        self.heads = config.heads
        self.n_kv_heads = config.n_kv_heads  # for each k and v
        self.head_dim = config.n_embd // config.heads


        self.w_q  = nn.Linear(config.n_embd, config.heads * self.head_dim) # n_embd -> q_heads * head_dim
        self.w_kv = nn.Linear(config.n_embd, 2*config.n_kv_heads * self.head_dim) # n_embd -> 2*kv_heads * head_dim

        self.register_buffer('mask',torch.tril(torch.ones((config.max_seq_len,config.max_seq_len))))

        self.proj = nn.Linear(config.n_embd , config.n_embd)


    def forward(self,x,start_pos,freq_complex):

        B,T,C = x.shape

        # queries
        q = self.w_q(x)  # [B,T,C] ; C = q_heads*head_dim
        q = q.reshape(B,T,self.heads,self.head_dim).transpose(1,2)  # [B,heads,T,head_dim]

        # key values
        k,v = self.w_kv(x).chunk(2,dim=-1) # [B,T, n_kv_heads*head_dim]  - for each

        k = k.reshape(B,T,self.n_kv_heads,self.head_dim).transpose(1,2) # [B,n_kv_heads,T,head_dim]
        v = v.reshape(B,T,self.n_kv_heads,self.head_dim).transpose(1,2) # [B,n_kv_heads,T,head_dim]


        # RoPE to queries and keys
        q_rotated = rope(freq_complex,q)
        k_rotated = rope(freq_complex,k)

        # repeat k for q
        n_group = self.heads//self.n_kv_heads   # for one key value -> how many queries
        k_repeat = self._repeat_kv_for_gqa(n_group,k_rotated)     # [B,heads,T,head_dim] ; heads = n_group * n_kv_heads

        # attention score
        weights = ( q_rotated * k_repeat.transpose(-1,-2) ) / (self.head_dim**0.5)

        # masking weights and normalize   (Autoregressiveness)
        weights_masked = weights.masked_fill(self.mask[:T,:T]==0,float('-inf'))
        weights_normalized = torch.softmax(weights_masked,dim = -1)          # -inf -> 0

        # for each query head repeated values (v_repeat)
        v_repeat = self._repeat_kv_for_gqa(n_group,v)             # [B,heads,T,head_dim]

        # apply Attention scores
        y = weights_normalized @ v_repeat                   # [B,heads,T,head_dim]
        y = y.transpose(1,2).reshape(B,T,C)      # [B,T,heads,head_dim] -> [B,T,C]

        # linear projection
        out = self.proj(y)                       # [B,T,C]

        return out





    def _repeat_kv_for_gqa(self,n_group,x):

        if n_group == 1 :
            return x

        B,heads,T,C = x.shape

        return x.unsqueeze(2).expand(-1,-1,n_group,-1,-1).reshape(B,-1,T,C)









class LlamaDecorderLayer(nn.Module):

    def _init__(self,config):
        super().__init__()

        self.config = config

        self.input_layernorm = LlamaRMSNorm(config._n_embd)

        self.self_attn = LlamaAttention(config)

        self.post_attention_layernorm = LlamaRMSNorm(config._n_embd)

        self.mlp = LlamaMLP(config)


    def forward(self,embds,start_pos, freqs_complex):

        out = embds + self.attn(self.input_layernorm(embds), start_pos,freqs_complex)

        out = out + self.mlp(self.post_attention_layernorm(out))

        return out












class LlamaModel(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size,config.n_embd)

        self.layers = nn.ModuleList([LlamaDecorderLayer(config) for _ in range(config.n_layer)])

        self.norm = LlamaRMSNorm(config._n_embd)

        # for RoPE in attn
        self.freqs_complex = precompute_theta_pos_frequencies(self.config.n_embd // self.config.n_heads, self.config.max_seq_len , device=self.config.device)


    def forward(self,tokens,start_pos):

        B,T = tokens.shape

        embds = self.embeddings(tokens)

        # for RoPE in attn
        freqs_complex = self.freqs_complex[start_pos:start_pos + T]

        out = embds
        for layer in self.layers:
            out = layer(out,start_pos, freqs_complex)

        out_norm = self.norm(out)

        return out_norm












class LlamaForCausalLM(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.n_embd,config.vocab_size)


    def forward(self,tokens,start_pos):

        out = self.model(tokens,start_pos)

        logits = self.lm_head(out)

        return logits
