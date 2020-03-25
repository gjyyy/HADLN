

from keras import backend as K
from keras.engine.topology import Layer

class Self_Attention(Layer):

    def __init__(self, output_dim=256, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.gamma = self.add_weight(name='gamma',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.gamma[0])
        WK = K.dot(x, self.gamma[1])
        WV = K.dot(x, self.gamma[2])


        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = K.softmax(QK)

        V = K.batch_dot(QK, WV)

        return V

