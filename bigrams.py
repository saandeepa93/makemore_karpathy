from imports import *

import torch 

import matplotlib.pyplot as plt

if __name__ == "__main__":
  words = open("./data/names.txt", "r").read().splitlines()

  # Pairs of characters for every word
  b_dict = {}
  for word in words:
    word_lst = ["<S>"] + list(word) + ["<E>"]
    for ch1, ch2 in zip(word_lst, word_lst[1:]):
      bigrams = (ch1, ch2)
      b_dict[bigrams] = b_dict.get(bigrams, 0) + 1

  sorted_tuple = sorted(b_dict.items(), key=lambda kv: -kv[1])

  # Use Tensor format to store this data instead of dictionary
  # NOTE: <S> is replaced by a . to make things simpler
  unique_chs = sorted(list(set(''.join(words))))
  l = len(unique_chs)
  stoi = {s:i+1 for i,s in enumerate(unique_chs)}
  # stoi["<S>"] = 26
  # stoi["<E>"] = 27
  stoi['.'] = 0
  itos = {i:s for s,i in stoi.items()}

  b_tensor = torch.zeros((l+1, l+1), dtype=torch.int32)
  for word in words:
    word_lst = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(word_lst, word_lst[1:]):
      b_tensor[stoi[ch1], stoi[ch2]] += 1

  plt.figure(figsize=(16, 16))
  plt.imshow(b_tensor)
  for i in range(b_tensor.size(0)):
    for j in range(b_tensor.size(1)):
      chs = itos[i] + itos[j]
      plt.text(j, i, chs, ha="center", va="bottom", color="gray")
      plt.text(j, i, b_tensor[i,j].item(), ha="center", va="top", color="gray")
  plt.axis("off")
  plt.savefig("./data/bigrams.png")


  # CREATE Bigram model 
  # NOTE: Removed inefficiency with normalization part
  # NOTE: Lookup Broadcasting, V. Important
  P = b_tensor.float()
  P /= P.sum(dim=1, keepdim=True)
  g = torch.Generator().manual_seed(42)
  for i in range(5):
    idx = 0
    gen_names = []
    while True:
      p = P[idx]
      idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
      gen_names.append(itos[idx])

      if idx == 0:
        break
  
    print("".join(gen_names))

  # COMPUTE loss function
  # NOTE: NLL Training Loss is the loss computed by adding log-probs of bigrams in training data.
  log_prob = 0.0
  ctr = 0
  for word in words:
    word_lst = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(word_lst, word_lst[1:]):
      idx1 = stoi[ch1]
      idx2 = stoi[ch2]
      prob = P[idx1, idx2]
      log_prob += prob.log()
      ctr+=1

  print(f"{log_prob/ctr=}")
  


