
from keras import backend as K
from keras.engine.topology import Layer


class attention_sum(Layer):

    def __init__(self,output_dim=256, **kwargs):
        self.output_dim = output_dim
        super(attention_sum, self).__init__(**kwargs)


    def call(self,layer):
        
        return K.sum(layer,axis=1)

