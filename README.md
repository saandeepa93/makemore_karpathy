## Repo for learning Karpathy's makemore series. Includes the following

  * `micrograd.ipynb` - Simulates Torch's autograd functionality.
  * `bigrams.py` - Simple bigram model. Non learnable
  * `mlp.ipynb` - Introducing MLPs to bigrams.
  * `activations_BN.ipynb` - Training stats to monitor such as activation distribution, BN etc.
  * `backprob_ninja.ipynb` - Manual backpropagation of a 2-layer MLP with BN.
  * `wavenet.ipynb` - Wavenet
  * `nanogpt` - Simplest GPT from scratch. 
  * `mingpt` - GPT2 124M model from scratch
  * `minbpe` - Karpathy's BPE tokenizer implementation. Has training module.

*The code follows the tutorial but retains to some of my common practices like `YACS` configuration, and `einops`, `rearrage` libraries for tensor manipulation etc.*