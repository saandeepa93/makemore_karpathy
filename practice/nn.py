from imports import *

import torch 
import torch.nn.functional as F

import matplotlib.pyplot as plt

if __name__ == "__main__":

  device = torch.device("cpu")

  words = open("./data/names.txt", "r").read().splitlines()

  unique_chars = sorted(list(set(''.join(words))))

  stoi = {s:i+1 for i,s in enumerate(unique_chars)}
  stoi['.'] = 0
  itos = {i:s for s,i in stoi.items()}

  # BUILD NN INPUT
  xs, ys = [], []
  for word in words:
    word_lst = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(word_lst, word_lst[1:]):
      idx1 = stoi[ch1]
      idx2 = stoi[ch2]
      xs.append(idx1)
      ys.append(idx2)
  
  xs = torch.tensor(xs)
  ys = torch.tensor(ys)

  g = torch.Generator().manual_seed(214783647)
  W = torch.randn((27, 27), generator=g, requires_grad=True).to(device)

  num = xs.nelement()
  ic(num)
  for i in range(100):
    
    xenc = F.one_hot(xs, num_classes=27).float().to(device)
    logits = xenc @ W
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    ic(loss.item())

    W.grad = None
    loss.backward()
    W.data += -50. * W.grad

  
  # SAMPLING WORDS FROM NN
  for i in range(10):
    ix = 0 
    out = []
    while True:
      xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
      logits = xenc @ W
      counts = logits.exp()
      probs = counts/counts.sum(1, keepdim=True)

      ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
      out.append(itos[ix])

      if ix == 0:
        break
    
    print("".join(out))


  e()
