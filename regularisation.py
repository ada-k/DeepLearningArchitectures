import tensorflow as tf
from keras import regularizers
from keras.layers.core import Dropout


# L2, L1
tf.keras.layers.Dense(3, kernel_regularizer=regularizers.l2(.01))
tf.keras.layers.Dense(3, kernel_regularizer=regularizers.l1(.01))

# Droput
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        output_dim=2, 
        input_dim=input_num_units,
        activation='relu'),

    Dropout(0.25), tf.keras.layers.Dense( #.25 = P(Dropping)
    output_dim=2, 
    input_dim=hidden5_num_units, 
    activation='softmax'),
 ])

