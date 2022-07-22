## DeepLearningArchitectures
Implementations of various deep learning architectures + extra theoretical information

### [CNNS](cnn.py)
**Feature Extraction Local connectivity:**
1. Apply a feature extraction filter == convolution layer.
2. Add bias to the filter function.
3. Activate with a non-linear function e.g ReLU for thresholding.
4. Pooling: dimesnionality reduction of generated feature maps from 1&2 above. Still maintains spatial invariance.
**classification**
5. Apply a fully connected layer that uses the high-level features from 1-2-3 to perform classification.