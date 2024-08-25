import torch 
from torch import nn, optim
from torch.nn import functional as F 
from einops import rearrange

from icecream import ic 
from sys import exit as e

torch.manual_seed(1337)
with open("./input.txt", "r", encoding='utf-8') as f:
  text = f.read()

# -----------------------------hyperparameters------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chars = sorted(list(set(text)))
vocab_size = len(chars)
batch_size= 64
n_epochs = 5000
block_size = 256
eval_interval = 1000
lr = 3e-4
n_embed = 384
dropout = 0.2
num_heads = 6
num_block_layers = 6
# -----------------------------hyperparameters------------------------


class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(n_embed, n_embed*4), 
      nn.ReLU(),
      nn.Linear(n_embed*4, n_embed), 
      nn.Dropout(dropout)
    )
  
  def forward(self, x):
    return self.net(x)

class MultiHead(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.head = nn.ModuleList(
      [Head(head_size) for _ in range(num_heads)]
    )
    self.proj = nn.Linear(head_size * num_heads, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.head], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class Head(nn.Module):
  def __init__(self, head_size=32) -> None:
    super().__init__()

    self.head_size = head_size
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)

    # wei = torch.einsum('btc,bck->btk', q, k.transpose(-1, -2))
    # wei = wei * C ** -0.5
    wei = q @ k.transpose(-2, -1) * C **-.5

    # Decoder block which doesnt let you see future tokens. Important for this problem since we are predicting next token
    wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    v = self.value(x)
    out = wei @ v
    return out

class Block(nn.Module):
  def __init__(self, num_head, n_embed):
    super().__init__()
    head_size = n_embed // num_head
    self.sa = MultiHead(num_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
  
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    # self.sa_head = MultiHead(4, n_embed//4)
    # self.ffwd = FeedForward(n_embed)
    # self.blocks = nn.Sequential(
    #   Block(num_head=num_heads, n_embed=n_embed),
    #   Block(num_head=num_heads, n_embed=n_embed),
    #   Block(num_head=num_heads, n_embed=n_embed), 
    #   nn.LayerNorm(n_embed)
    # )

    self.blocks = nn.Sequential(
      *[Block(num_head=num_heads, n_embed=n_embed) for _ in range(num_block_layers)]
    )
    self.ln = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
  
  def forward(self, idx, target=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))

    x = tok_emb + pos_emb
    x = self.blocks(x)
    logits = self.lm_head(x)

    if target is not None:
      logits = rearrange(logits, 'b t c -> (b t) c')
      target = rearrange(target, 'b t -> (b t)')
      loss = F.cross_entropy(logits, target)
    else:
      loss = None
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat([idx, idx_next], dim=1)
    return idx


@torch.no_grad()
def estimate_loss(model):
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_interval)
    for k in range(eval_interval):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss
    out[split] = losses.mean()
  model.train()
  return out

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data)-block_size, size=(batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x = x.to(device)
  y = y.to(device)
  return x, y


if __name__ == '__main__':
  stoi = {ch:i for i, ch in enumerate(chars)}
  itos = {i:ch for i, ch in enumerate(chars)}
  encode = lambda x: [stoi[c] for c in x]
  decode = lambda x: ''.join([itos[i] for i in x])

  data = torch.tensor(encode(text), dtype=torch.long)

  t = int(0.9 * data.shape[0])
  train_data = data[:t]
  val_data = data[t:]


  model = BigramLanguageModel()
  model = model.to(device)

  optimizer = optim.AdamW(model.parameters(), lr=lr)

  for epoch in range(n_epochs+1):

    if epoch% eval_interval == 0:
      loss = estimate_loss(model)
      print(f"At epoch: {epoch}: train loss: {loss['train']:.4f}, val loss: {loss['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  print("IFNERENCE")
  idx = torch.zeros((1, 1), dtype=torch.long, device=device)
  ypred = ''.join(decode(model.generate(idx, max_new_tokens=2000)[0].cpu().tolist()))
  print(ypred)