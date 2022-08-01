## DeepLearningArchitectures
Implementations of various deep learning architectures + extra theoretical information

### 1. [Convolution Neural Nets - CNN](cnn.py)
**Feature Extraction Local connectivity:**
1. Apply a feature extraction filter == convolution layer.
2. Add bias to the filter function.
3. Activate with a non-linear function e.g ReLU for thresholding.
4. Pooling: dimesnionality reduction of generated feature maps from 1&2 above. Still maintains spatial invariance.

**Classification:**
1. Apply a fully connected layer that uses the high-level features from 1-2-3 to perform classification.

Other potential applications; to achieve, you only modify the fully-connected layer bit == step1 under classification.
- Segmentation.
- Object Detection.
- Regression.
- Probabilistic Control.
- Autonomous navigation.



### 2. [Deep Generative Modeling]()
**Applications:**
- Density Estimation.
- Sample Generation.

#### 2.1 [Autoencoders and Variational AE](vae.py)
- Input data X is encoded(compressed, self-encoded, auto encoded) to  a lower dimensional latent space Z then model(decoder network) learns from the Z to reconstruct x as x hat.
- Loss function == sq of difference between pixel by pixel difference between x and x hat.
- VAEs introduces a stochasti process/randomness/probability aspect by calculating mean and SD from which the latent space features Z are sampled from.
- VAEs loss = reconstruction loss + regularisation term(enforcess continuity and completeness).
- Regularisation(D) = distance between 2 distributions (techniques like KL-divergence can be used to qunatify)
![image](https://user-images.githubusercontent.com/50487929/180368201-c78caf23-c2f0-4afb-8133-5ed6bc4cc3c9.png) (prior is introduced)
- To counter the challenge of backpropagation cause of stochasticty, reparametrisation is introduced == calculating a fixed vector of means and SDs thus driving away the probabilistic nature away from means and SDs of z.



#### 2.2 [Generative Adversarial Networks - GANS](gans.py)
- Generators starts from noise(z==latent space) then imitates input data based on this.
- Discriminator tries to identify real data from fakes created by the generator.(minimises probability of fake)
- The process is iterative till the discriminator produces the highest probability that the generated data is real.
- **G** == global optimum = produce the true data distribution == minmax

### 3. [Deep Sequence Modeling]() 
- output at time t = a function of the input at t and past memory at time t-1.

#### 3.1 [Recurrent Neural Nets - RNN](rnn.py)

**Steps:**
1. Initialise the weight matrices and the hidden state to 0.
2. Defining the core function: the forward pass through:
  1. Update of the hidden state(input + previous state).
  2. Output computation to generate output and new hidden state.
  
**Design Criteria:**
- Handle sequences of variable length.
- Track longterm dependencies.
- Maintain info about order.
- Share parameters across the sequence.


#### 3.2 [LSTM](lstm.py)
Concepts:
1. Maintains cell state.
2. Has gates that control info flow: eliminates irrelevantt, keeps relevant.
3. Backpropagation through time with partially uninterrupted gradient flow.

#### 3.3 [Transformers](transformers.py)
...

### 4. [Reinforcement Learning](rl.py)
...

### 5. [Transfer Learning](tl.py)
...


### 0. [Elements of Neural Nets Architecture](#)
#### Layers:
- **Input layer** - Receives raw input data and passes the info to follow up layers - hidden. No computation occurs here.
- **Hidden layer** - An abstraction of the fully formed network. Performs computation on features from input layer(or another hidden layer) and passes result to output laye r(or another hidden layer). PS: They all use the same activation function.
- **Output layer** - Final layer of the network hilding final value/prediction/output of the network. Has a different activation function depending on the goal.


#### Feedforward Network
*Connectivity*: Forward flow of information: Input used to calculate intermediate function in the hiddent layer to generate an output.
1. Multiplication of input with neuron weights.
2. Addition of bias.
3. Passing output through an activation function.
4. Output layer.

#### Backpropagation
- Repeated adjusting of network weights and bias to minimise the cost/loss function based on the previous epoch.
- The level of adjustment is determined by the gradients of the cost function with respect to those parameters.
- *Connectivity* - calculating gradients of the loss/error function, then updating existing parameters in response to the gradients

#### Weights and Bias
- **Weight** - Learnable. Transforms input data within the hidden layers. 
- **Bias** - Learnable. Represents how off the predictions are from the expected values. == error term but not cost function.


#### [Activation Functions](activation.py)
- Root purpose: not all data is linear. Performing a simple calculation of weights and bias would otherwise be a normal layered linear regression.

**For Hidden Layers**:
1. *Rectified Linear Activation (ReLU)*
- Function is linear for values > 0 but still non linear since all -ve values are tranformed to 0
```
# g(z) = max{0, z}
if input > 0:
	return input
else:
	return 0
```

2. *Logistic (Sigmoid)*
- Takes real values as input and outputs them in the range(0,1)
```
g(x) = 1.0 / (1.0 + e^-x)
```
- Prone to problem of vanishing gradients.

3. *Hyperbolic Tangent (Tanh)*
- Takes real value inputs and outputs them in the range(-1, 1)
- Takes same shape as a sigmoid.
```
g(x) = (e^x – e^-x) / (e^x + e^-x)
```
- Prone to problem of vanishing gradients.


**For Output Layers**:
1. *Linear*
- == no activation/identity. Returns output as is.

2. *Logistic (Sigmoid)*

3. *Softmax*
- Outputs a vector of values that sum up to 1.0. Like probabilities.

```
g(x) = e^x / sum(e^x)
```


#### Optimisation Techniques
##### Gradient Descent
##### Stochastic Gradient Descent (SGD)
##### Mini-Batch Stochastic Gradient Descent (MB — SGD)
##### SGD with Momentum
##### Nesterov Accelerated Gradient (NAG)
##### Adaptive Gradient (AdaGrad)
##### AdaDelta
##### RMSProp
##### Adam
##### Nadam

#### Gradient Issues:
- *Vanishing gradients* - 
- *Exploding gradients* - 

#### Convergence:
- **Global optimum** - 
- **Local Minima** - 

#### Regularisation Terms


### 0.1 [Prepping Neural Nets INput data](prep.py)
- **Text** - 
- **Images** - 
- **Videos** - 

