from imports import * 
import torch 
from torch.nn import functional as F

import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (20, 16)


def build_dataset(words, stoi):
  block_size = 3
  X = []
  y = []
  for word in words:
    context = [0] * block_size
    for ch in word:
      X.append(context)
      y.append(stoi[ch])

      context = context[1:] + [stoi[ch]]
  X = torch.tensor(X)
  y = torch.tensor(y)
  return X, y

if __name__ == "__main__":
  words = open("./data/names.txt", "r").read().splitlines()
  unique_chars = sorted(list(set(''.join(words))))

  stoi = {s: i+1 for i, s in enumerate(unique_chars)}
  stoi['.'] = 0
  itos = {i: s for s, i in stoi.items()}

  random.seed(42)
  random.shuffle(words)
  t1 = int(0.8 * len(words))
  t2 = int(0.9 * len(words))

  X_train, y_train = build_dataset(words[:t1], stoi)
  X_val, y_val = build_dataset(words[t1:t2], stoi)
  X_test, y_test = build_dataset(words[t2:], stoi)

  ic(X_train.shape, y_train.shape)
  ic(X_val.shape, y_val.shape)
  ic(X_test.shape, y_test.shape)
  # BUILD DATASET FOR MLP
  C = torch.randn((27, 10))


  g = torch.Generator().manual_seed(2147483647)
  W1 = torch.randn((30, 100), requires_grad=True, generator = g)
  b1 = torch.randn((100), requires_grad=True, generator=g)
  W2 = torch.randn((100, 27), requires_grad=True, generator=g)
  b2 = torch.randn((27), requires_grad=True, generator=g)

  parameters = [W1, b1, W2, b2]
  n = sum(p.nelement() for p in parameters)
  ic(f"{n=}")

  for p in parameters:
    p.requires_grad=True

  total_loss = []
  steps = []
  for epoch in range(50000):
    idx = torch.randint(0, X_train.shape[0], (256,))

    # FORWARD PASS
    # NOTE: Notice that a (27, 2) is indexed by a (D, 3) integer tensor
    emb = C[X_train[idx]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    # NOTE: CE is more efficient in pytorch. 
    # NOTE: CE is numerically well-behaved in torch. logits - torch.max(logits) 
    loss = F.cross_entropy(logits, y_train[idx])

    # BACKWARD PASS
    for p in parameters:
      p.grad = None
    loss.backward()
    for p in parameters:
      p.data += -0.01 * p.grad 

    with torch.no_grad():
      total_loss.append(loss.item())
      steps.append(epoch)

  plt.plot(steps, total_loss)
  plt.savefig("./data/loss_mlp.png")
  ic(sum(total_loss)/len(total_loss))


  with torch.no_grad():
    emb_val = C[X_val]
    h = torch.tanh(emb_val.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    val_loss = F.cross_entropy(logits, y_val)
    ic(val_loss)


