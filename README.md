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

