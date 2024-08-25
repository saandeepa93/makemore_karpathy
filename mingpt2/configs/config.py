from yacs.config import CfgNode as CN

_C = CN()

# PATHS

_C.DATASET = CN()
_C.DATASET.BLOCK_SIZE = 4
_C.DATASET.VOCAB_SIZE = 0

_C.TRAINING = CN()
_C.TRAINING.ITER=100
_C.TRAINING.LR=1e-4
_C.TRAINING.WT_DECAY=1e-5
_C.TRAINING.BATCH_SIZE=256
_C.TRAINING.BETA1=0.9
_C.TRAINING.BETA2=0.99
_C.TRAINING.MAX_LR=3e-4
_C.TRAINING.MIN_LR_PERC=0.1
_C.TRAINING.MAX_STEPS=50
_C.TRAINING.WARMUP_STEPS=50
_C.TRAINING.DISTRIBUTED=False

_C.GPT = CN()
_C.GPT.N_LAYER = 1
_C.GPT.N_HEAD = 1
_C.GPT.N_EMBD = 1


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()