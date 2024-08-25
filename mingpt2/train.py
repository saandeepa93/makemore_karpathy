# torch and variants
import torch 
from torch import nn, optim
from torch.nn import functional as F
from einops import rearrange
import numpy as np

# ddp
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# huggingface
from transformers import GPT2LMHeadModel

# openai
import tiktoken

# utils
import glob
from dataclasses import dataclass
import math
import os

# hebugger helpers
import time
import code
from icecream import ic 
from sys import exit as e

# external imports
from configs import get_cfg_defaults

# ----------------------------CONFIGS-------------------------------
@dataclass
class GPTConfig:
  block_size: int = 256 
  vocab_size: int = 65 
  n_layer: int = 6 
  n_head: int = 6 
  n_embd: int = 384
# ----------------------------CONFIGS-------------------------------



# ----------------------------DataLoader-------------------------------
def load_tokens(filename):
  arr = np.load(filename)
  arr = arr.astype(np.int16)
  ptt = torch.tensor(arr, dtype=torch.long)
  return ptt

class MyLoader:
  def __init__(self, config, mode, cur_rank, world_size):

    self.B, self.T = config.TRAINING.BATCH_SIZE, config.DATASET.BLOCK_SIZE
    self.cur_rank = cur_rank
    self.world_size = world_size

    self.shard_list = sorted(glob.glob(os.path.join(cfg.PATHS.DATA_ROOT, 'edu_fineweb10B', f'*{mode}*.npy')))
    self.curr_shard = 0 

    self.tokens = load_tokens(self.shard_list[self.curr_shard])
    self.current_pos = self.B * self.T * cur_rank
    print(f"Loaded {len(self.tokens)} tokens")
    self.reset()

  def reset(self):
    self.curr_shard = 0 
    self.tokens = load_tokens(self.shard_list[self.curr_shard])
    self.current_pos = self.B * self.T * self.cur_rank
  
  def get_batch(self):
    buffer = self.tokens[self.current_pos: self.current_pos+self.B*self.T+1]
    x = buffer[:-1].view(self.B, self.T)
    y = buffer[1:].view(self.B, self.T)

    self.current_pos += self.B * self.T * self.world_size
    if self.current_pos + (self.B * self.T * self.world_size +1 )> len(self.tokens):
      self.curr_shard = (self.curr_shard + 1) % len(self.shard_list)
      self.tokens = load_tokens(self.shard_list[self.curr_shard])
      self.current_pos = self.B * self.T * self.cur_rank
    
    return x, y

# ----------------------------DataLoader-------------------------------


# ----------------------------GPT-------------------------------

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config 
    assert config.GPT.N_EMBD % config.GPT.N_HEAD == 0, "number of heads should be divisble by embed dims"

    self.c_attn = nn.Linear(config.GPT.N_EMBD, config.GPT.N_EMBD * 3)
    self.c_proj = nn.Linear(config.GPT.N_EMBD, config.GPT.N_EMBD)
    self.c_proj.NANO_GPT_SCALE_INIT = 1

    trils = torch.tril(torch.ones((config.DATASET.BLOCK_SIZE, config.DATASET.BLOCK_SIZE)).\
                       view(1, 1, config.DATASET.BLOCK_SIZE, config.DATASET.BLOCK_SIZE))
    self.register_buffer('bias', trils)

    self.n_head = config.GPT.N_HEAD
    self.n_embd = config.GPT.N_EMBD

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=-1)
    q = rearrange(q, 'b t (h d) -> b h t d', b=B, t=T, h=self.n_head)
    k = rearrange(k, 'b t (h d) -> b h t d', b=B, t=T, h=self.n_head)
    v = rearrange(v, 'b t (h d) -> b h t d', b=B, t=T, h=self.n_head)

    # attn = torch.einsum('bhtd, bhdk -> bhtk', q, k.permute(0, 1, 3, 2)) * (1/math.sqrt(k.size(-1)))
    # attn = attn.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
    # attn = F.softmax(attn, dim=-1)
    # y = torch.einsum('bhtk, bhkd -> bhtd', attn, v)
    
    # NOTE: Fused attention kernel for speedup
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = rearrange(y, 'b h t d -> b t (h d)')

    y = self.c_proj(y)

    return y




class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.config = config 
    self.c_fc = nn.Linear(config.GPT.N_EMBD, 4 * config.GPT.N_EMBD)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.GPT.N_EMBD, config.GPT.N_EMBD)
    self.c_proj.NANO_GPT_SCALE_INIT = 1
  
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.mlp = MLP(config)
    self.attn  = CausalSelfAttention(config)
    self.ln_1 = nn.LayerNorm(config.GPT.N_EMBD)
    self.ln_2 = nn.LayerNorm(config.GPT.N_EMBD)
  
  def forward(self, x):
    x = x + self.attn(self.ln_1(x)) 
    x = x + self.mlp(self.ln_2(x))
    return x


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.DATASET.VOCAB_SIZE, config.GPT.N_EMBD),
        wpe = nn.Embedding(config.DATASET.BLOCK_SIZE, config.GPT.N_EMBD),
        h = nn.ModuleList([Block(config) for _ in range(config.GPT.N_LAYER)]),
        ln_f = nn.LayerNorm(config.GPT.N_EMBD)
      )
    )

    self.lm_head = nn.Linear(config.GPT.N_EMBD, config.DATASET.VOCAB_SIZE, bias=False)

    # NOTE: Weight sharing. check attention paper 
    self.transformer.wte.weight = self.lm_head.weight

    # NOTE: following gpt2 intiialization
    self.apply(self._init_weights)


  def configure_optimizer(self, cfg, device):
    param_dict = {n:p for n,p in self.named_parameters()}
    param_dict = {n:p for n, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
    non_decay_params = [p for n,p in param_dict.items() if p.dim() < 2]

    num_decayed = sum(p.numel() for p in decay_params)
    num_nondecayed = sum(p.numel() for p in non_decay_params)

    print(f'Number of decayed params: {num_decayed}')
    print(f'Number of non-decayed params: {num_nondecayed}')
    
    param_groups = [
      {'params': decay_params, 'weight_decay': cfg.TRAINING.WT_DECAY},
      {'params': non_decay_params, 'weight_decay': 0.0}
    ]

    # NOTE: For some reason fused slows down training for me. Defaulted to False
    fused = True if torch.cuda.is_available() else False
    optimizer = optim.AdamW(param_groups, lr=cfg.TRAINING.LR, betas=(cfg.TRAINING.BETA1, cfg.TRAINING.BETA2), eps=1e-8, fused=False)
    return optimizer
  
  def _init_weights(self, module):
    std = 0.02

    # NOTE: any residual block incurs an increased variance due to "addition" of two normal random variables. 
    #       To maintain variance at 1, the weights
    #       right before addition are scaled by sqrt(2*nlayer) times. 2 because each layer has 2 residual operations. 
    # NOTE: SEE JUPYTER NOTEBOOK FOR INTUITION
    
    if hasattr(module, "NANO_GPT_SCALE_INIT"):
      std *= (2 * self.config.GPT.N_LAYER) ** -.5 
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0., std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0., std=0.02)
  
  def forward(self, idx, target=None):
    B, T = idx.size()

    assert T <= self.config.DATASET.BLOCK_SIZE, "context length exceeded max block size"

    pos = torch.arange(0, T, device=idx.device)
    pos_emb = self.transformer.wpe(pos)

    print(idx.max(), idx.min(), self.transformer.wte.weight.shape)
    tok_emb = self.transformer.wte(idx)

    x = pos_emb + tok_emb
    for block in self.transformer.h:
      x = block(x)
    
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)

    loss = None 
    if target is not None:
      loss = F.cross_entropy(logits.view(-1, self.config.DATASET.VOCAB_SIZE), target.view(-1))

    return logits, loss

     

  @classmethod
  def from_pretrained(cls, model_type):
      """Loads pretrained GPT-2 model weights from huggingface"""
      assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
      from transformers import GPT2LMHeadModel
      print("loading weights from pretrained gpt: %s" % model_type)

      # n_layer, n_head and n_embd are determined from model_type
      config_args = {
          'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
          'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
          'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
          'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
      }[model_type]
      config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
      config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
      # create a from-scratch initialized minGPT model
      config = GPTConfig(**config_args)
      # model = GPT(config)
      model = GPT(cfg)
      sd = model.state_dict()
      sd_keys = sd.keys()
      sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

      # init a huggingface/transformers model
      model_hf = GPT2LMHeadModel.from_pretrained(model_type)
      sd_hf = model_hf.state_dict()

      # copy while ensuring all of the parameters are aligned and match in names and shapes
      sd_keys_hf = sd_hf.keys()
      sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
      sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
      transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
      # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
      # this means that we have to transpose these weights when we import them
      assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
      for k in sd_keys_hf:
          if any(k.endswith(w) for w in transposed):
              # special treatment for the Conv1D weights we need to transpose
              assert sd_hf[k].shape[::-1] == sd[k].shape
              with torch.no_grad():
                  sd[k].copy_(sd_hf[k].t())
          else:
              # vanilla copy over the other parameters
              assert sd_hf[k].shape == sd[k].shape
              with torch.no_grad():
                  sd[k].copy_(sd_hf[k])

      return model
# ----------------------------GPT-------------------------------


def get_lr(cfg, epoch):
  if epoch<cfg.TRAINING.WARMUP_STEPS:
    return cfg.TRAINING.MAX_LR * (epoch+1)/cfg.TRAINING.WARMUP_STEPS
  if epoch > cfg.TRAINING.ITER:
    return cfg.TRAINING.MIN_LR_PERC * cfg.TRAINING.MAX_LR
  
  decay_ratio = (epoch - cfg.TRAINING.WARMUP_STEPS)/(cfg.TRAINING.ITER - cfg.TRAINING.WARMUP_STEPS)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
  return cfg.TRAINING.MIN_LR_PERC * cfg.TRAINING.MAX_LR + coeff * \
    (cfg.TRAINING.MAX_LR - (cfg.TRAINING.MIN_LR_PERC * cfg.TRAINING.MAX_LR))


if __name__ == '__main__':

  torch.manual_seed(1337)
  torch.cuda.manual_seed(1337)

  cfg = get_cfg_defaults()
  cfg.merge_from_file("./configs/experiments/gpt2.yaml")

  # DISTRIBUTED USAGE
  if cfg.TRAINING.DISTRIBUTED:
    assert torch.cuda.is_available(), "No CUDA detected"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
  else:
    ddp_rank = 0 
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device_cap = torch.cuda.get_device_capability()
  device = torch.device('cpu')
  
  total_batch_size = 2**18 # ~0.5M tokens 
  cur_batch = cfg.TRAINING.BATCH_SIZE * cfg.DATASET.BLOCK_SIZE * ddp_world_size
  grad_accum_steps = total_batch_size//(cur_batch * ddp_world_size) # x ddp_world_size because 1 GPU -> 0.5M tokens. 2 GPU -> 0.25M tokens each 

  if master_process:
    print(f"Desired batch size: {total_batch_size}")
    print(f"Micro batch size: {cur_batch}")
    print(f"Therefore, update gradients after every {grad_accum_steps} steps")

  # NOTE: wont work cos of cur GPU
  torch.set_float32_matmul_precision("high")

  # DATA PREPARATION
  trainloader = MyLoader(cfg, "train", ddp_local_rank, ddp_world_size)
  valloader = MyLoader(cfg, "val", ddp_local_rank, ddp_world_size)

  # MODEL LOADING
  # NOTE: Run unit test case generation. Fixed bugs in attention computation
  # model = GPT.from_pretrained('gpt2')

  # NOTE: random model text generation will still give words that makes sense. That is because of openai tokenizer that has learned meaningful subwords. 
  model = GPT(cfg)
  model.to(device)
  # model = torch.compile(model)

  enc = tiktoken.get_encoding('gpt2')


  if cfg.TRAINING.DISTRIBUTED:
    model = DDP(model, device_ids = [ddp_local_rank])
  
  raw_model = model.module if cfg.TRAINING.DISTRIBUTED else model
  if master_process:
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
  
  # OPTIMIZER
  # optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR, betas=(cfg.TRAINING.BETA1, cfg.TRAINING.BETA2))
  optimizer = raw_model.configure_optimizer(cfg, device)
  
  # start training
  for epoch in range(cfg.TRAINING.ITER):
    start = time.time()

    # Validation
    if epoch%100 ==0 :
      model.eval()
      valloader.reset()
      with torch.no_grad():
        val_loss_accum = 0.
        val_num_steps = 20 
        for micro_step in range(val_num_steps):
          x, y = valloader.get_batch()
          x, y = x.to(device), y.to(device)

          logits, loss = model(x, y)

          loss /= val_num_steps
          val_loss_accum += loss.detach()
      
      if cfg.TRAINING.DISTRIBUTED:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
      if master_process:
        print(f"Validation Loss: {val_loss_accum.item(): .4f}")


    # Sampling 
    if epoch > 0 and epoch %100 ==0 :
      num_sequences = 5
      max_length = 30 
      tokens = enc.encode("Hello, I'm a language model,")
      tokens = torch.tensor(tokens, dtype=torch.long)
      tokens = tokens.view(1, -1).repeat(num_sequences, 1)
      xgen = tokens.to(device)

      rng = torch.Generator(device=device)
      rng.manual_seed(42 + ddp_local_rank)

      while xgen.size(1) < max_length:
        with torch.no_grad():
          logits, _ = model(xgen)

          probs = F.softmax(logits, dim=-1)
          topk_probs, topk_ind = torch.topk(probs, 50, dim=-1)
          idx = torch.multinomial(topk_probs, 1, generator=rng)

          xnew = torch.gather(topk_ind, dim=-1, index=idx)
          xgen = torch.cat((xgen, xnew), dim=1)
      for i in range(num_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"GPU {ddp_local_rank}, sample {i}: {decoded}")



    # Train
    model.train()
    optimizer.zero_grad()
    loss_accum = 0

    for micro_epoch in range(grad_accum_steps):
      x, y = trainloader.get_batch()
      x = x.to(device)
      y = y.to(device)
      

      # NOTE: my gpu does not support bfloats and tensorfloats
      # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

      logits, loss = model(x, y)

      loss = loss/grad_accum_steps # So that gradients accumulate on the normalized loss. Check ipynb file for intuition
      loss_accum += loss.detach()

      #NOTE: debugger
      # code.interact(local=locals())

      # NOTE: in DDP mode, synchronize gradients across GPUs/ranks only after reaching end of grad_accum_steps
      if cfg.TRAINING.DISTRIBUTED:
        model.require_backward_grad_sync = (micro_epoch == grad_accum_steps - 1)
      loss.backward()

    if cfg.TRAINING.DISTRIBUTED:
      dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # NOTE: Grad clipping
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # NOTE: LR Scheduling
    lr = get_lr(cfg, epoch)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize()
    t2 = time.time()
    total = (t2 - start)*1000
    token_per_sec = (trainloader.B * trainloader.T * grad_accum_steps * ddp_world_size)/(t2 - start) 

    if master_process:
      print(f"epoch: {epoch}, loss: {loss_accum.item(): .4f}, lr: {lr:.4e}, gradnorm: {norm:.4f}, time: {total: .4f}, tok/sec: {token_per_sec:.4f}")

  if cfg.TRAINING.DISTRIBUTED:
    destroy_process_group()

  # ic(logits.shape, loss.item())
  e()



  # model = GPT2LMHeadModel.from_pretrained('gpt2')
  # model = model.to(device)
  num_sequences = 5
  max_length = 30
  # generation
  while x.size(1) < max_length:
    with torch.no_grad():
      logits = model(x)
      ic(logits.shape)
      e()
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)

      top_k_probs, top_k_ind = torch.topk(probs, 50, dim=-1)

      ix = torch.multinomial(top_k_probs, 1)
      xcol = torch.gather(top_k_ind, dim=-1, index=ix)
      x = torch.cat((x, xcol), dim=1)

  
  for i in range(num_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
  print("Dint crash :")
