import tensorflow as tf

def create_model():

    model = tf.keras.Sequential([

        # feature extraction: layer1
        tf.keras.layers.Conv2D(32, filter=3, activation='relu'), #the output's == 32 feature maps
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2), #downsizing by 2

        # layer2: 2nd conv layer
        tf.keras.layers.Conv2D(64, filters=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # observe the trade off in increase of featuremaps (32 to 64) and reduction in input size
        # since we've succesfully downsampled our input after conv layer1, we can afford to increase
        # the resolution of our feature dimension

        # fully connected classifier
        tf.layers.Flatten(), #flatten our dense information
        tf.layers.Dense(1024, activation="relu"), # classifier layer
        tf.layers.Dense(10, activation="softmax") # number of outputs=10

    ])

    return model