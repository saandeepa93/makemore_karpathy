## **BatchNorm**

* Stablize the activations of NN
* Acts as a regularizer because you add some noise of other inputs in a batch to teh current input. In the form of mean and std.
* Keeps track of running mean and std for inference later on. Model eval will ensure this in pytorch 

## Init
* Best first loss case for NLL/CE is -log(1/k). To do this, use kaiming he init or uniform init