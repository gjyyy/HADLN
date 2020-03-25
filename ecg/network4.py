from keras import backend as K
import numpy as np
from selfattention import *
from keras.layers import LSTM,Bidirectional,Add,Multiply
from keras.layers.core import Dense, Activation
from keras.layers.core import Lambda
import tensorflow as tf

###blend-attention###


def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D
    layer = Conv1D(
        filters=num_filters,#16
        kernel_size=filter_length,#32
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer



def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:#下采样
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],#16
                    params["conv_num_filters_start"],#32
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters


def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer



def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('softmax')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])



def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')

    layer = add_conv_layers(inputs, **params)

    layer_bilstm = Bidirectional(LSTM(128 ,return_sequences=True),merge_mode='concat')(layer)



    if params.get('is_regular_conv', False):

        layer=add_conv_layers(inputs,**params)
    else:
        layer = add_resnet_layers(inputs, **params)

    att_lstm = Dense(256, activation='linear', use_bias=False)(layer_bilstm)
    att_cnn = Dense(256, activation='linear', use_bias=False)(layer)
    all_feat = Add()([att_cnn , att_lstm])
    activate_feat = Activation('tanh')(all_feat)
    unnorm_attention = Dense(256, activation='linear', use_bias=False)(activate_feat)

    norm_attention =  Activation('softmax')(unnorm_attention)#(?,?,256)

    weighted_feat = Multiply()([norm_attention,layer])

    output = add_output_layer(weighted_feat, **params)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    if params.get("compile", True):
        add_compile(model, **params)
    return model
