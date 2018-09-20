from keras.layers import Dense, SeparableConv1D, LSTM
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization, Masking
from keras.layers import Lambda, Subtract, Concatenate, Reshape

import keras.backend as K


def dnn_build(embed_input1, embed_input2):
    mean = Lambda(lambda a: K.mean(a, axis=1), name='mean')
    da1 = Dense(200, activation='relu', name='encode1')
    da2 = Dense(200, activation='relu', name='encode2')
    dist = Lambda(lambda a: K.sum(K.square(a), axis=1))
    x = mean(embed_input1)
    x = da1(x)
    x = da2(x)
    y = mean(embed_input2)
    y = da1(y)
    y = da2(y)
    z = Subtract()([x, y])
    z = dist(z)
    return Reshape((1,))(z)


def dnn_cache(embed_input):
    mean = Lambda(lambda a: K.mean(a, axis=1), name='mean')
    da1 = Dense(200, activation='relu', name='encode1')
    da2 = Dense(200, activation='relu', name='encode2')
    x = mean(embed_input)
    x = da1(x)
    return da2(x)


def cnn_build(embed_input1, embed_input2):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    da = Dense(1, activation='sigmoid')
    x1 = ca1(embed_input1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalMaxPooling1D()(x1)
    x2 = ca2(embed_input1)
    x2 = BatchNormalization()(x2)
    x2 = GlobalMaxPooling1D()(x2)
    x3 = ca3(embed_input1)
    x3 = BatchNormalization()(x3)
    x3 = GlobalMaxPooling1D()(x3)
    x = Concatenate()([x1, x2, x3])
    y1 = ca1(embed_input2)
    y1 = BatchNormalization()(y1)
    y1 = GlobalMaxPooling1D()(y1)
    y2 = ca2(embed_input2)
    y2 = BatchNormalization()(y2)
    y2 = GlobalMaxPooling1D()(y2)
    y3 = ca3(embed_input2)
    y3 = BatchNormalization()(y3)
    y3 = GlobalMaxPooling1D()(y3)
    y = Concatenate()([y1, y2, y3])
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def cnn_cache(embed_input):
    pass


def rnn_build(embed_input1, embed_input2):
    ra = LSTM(200, activation='tanh')
    da = Dense(1, activation='sigmoid')
    x = Masking()(embed_input1)
    x = ra(x)
    y = Masking()(embed_input2)
    y = ra(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def rnn_cache(embed_input):
    pass
