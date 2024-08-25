# **Training caveats**

* The first CE loss should always be around $-\log (1/numlogits)$
* Adding normal distribution keeps increasing its variance. This happens in residual blocks
* Overfit on a single batch by trainign on multiple epochs.
* Unit test extensively for bugs. 
* Max out 1 gpu using batch size primarily.
* In Language model, initial stages of training gradients basically learns the most commonly occuring tokens/strings. It learns to ignore rare tokens. In the later stages of training is when the model learning more distinct information about differnt tokens
* `Weight_decay` basically helps the model to train acorss all weights. If a value of weight is too large, the model tends to focus on those more, so we need to pull those down using decay parameters.

# **Speedup**
* `torch.set_float32_matmul_precision("high")` : TensorFloat. Wont work for non-ampere GPUs
* `torch.compile` Compiles your NNs and finds operations that avoids multiple GPU chip <-> HBM movement of operations
* `Flash Attention` Calculate Attention under a fused kernel. Heavily memory architecture focused.
*  `nice numbers` Keep hyperparameters primarily at powers of 2
* Optimizers used `fused=True`. Basically performs optimization in a fused kernel. Increases processing time for my gpu
* `grad_accum` to imitate a bigger batch size by performing step only after accumulating gradients until specified bsize/

## **BatchNorm**

* Stablize the activations of NN
* Acts as a regularizer because you add some noise of other inputs in a batch to teh current input. In the form of mean and std.
* Keeps track of running mean and std for inference later on. Model eval will ensure this in pytorch 

## Init
* Best first loss case for NLL/CE is -log(1/k). To do this, use kaiming he init or uniform init