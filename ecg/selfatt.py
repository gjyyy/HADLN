

from keras import backend as K
from keras.engine.topology import Layer

class Self_Attention2(Layer):

    def __init__(self, output_dim=256, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention2, self).__init__(**kwargs)


    def call(self, x, y):

        W = K.batch_dot(x,y)

        return W

