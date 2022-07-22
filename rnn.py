import tensorflow as tf

class RNN(tf.keras.layers.Layer()):

    def __init__(self, rnn_units, input_dim, output_dim):
        super(RNN, self).__init__()

        # weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        # initialise hidden state to 0
        self.h = tf.zeros([rnn_units, 1])

    def main():

        # update the hidden state
        self.h = tf.math.tanh(self.W_hh * self.h * self.W_xh * x)

        # compute output
        output = self.W_hy * self.h

        return output, self.h



# """short form"""

tf.keras.layers.SimpleRNN(rnn_units)