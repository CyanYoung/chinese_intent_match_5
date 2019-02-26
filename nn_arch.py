from keras.layers import LSTM, Dense, Bidirectional, Dropout, Lambda
from keras.layers import Concatenate, Flatten, Reshape, Subtract, Multiply, Dot

import keras.backend as K
from keras.engine.topology import Layer


class Attend(Layer):
    def __init__(self, unit, **kwargs):
        self.unit = unit
        super(Attend, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.seq_len = input_shape[0][1]
        self.embed_len = input_shape[0][2]
        self.w = self.add_weight(name='w', shape=(self.embed_len * 2, self.unit),
                                 initializer='glorot_uniform')
        self.b1 = self.add_weight(name='b1', shape=(self.unit,),
                                  initializer='zeros')
        self.v = self.add_weight(name='v', shape=(self.unit, 1),
                                 initializer='glorot_uniform')
        self.b2 = self.add_weight(name='b2', shape=(1,), initializer='zeros')
        super(Attend, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        h1, h2 = x
        c = list()
        for i in range(self.seq_len):
            h2_i = K.repeat(h2[:, i, :], self.seq_len - 1)
            x = K.concatenate([h1, h2_i])
            p = K.tanh(K.dot(x, self.w) + self.b1)
            p = K.softmax(K.dot(p, self.v) + self.b2)
            p = K.squeeze(p, axis=-1)
            p = K.repeat(p, self.embed_len)
            p = K.permute_dimensions(p, (0, 2, 1))
            c_i = K.sum(p * h1, axis=1, keepdims=True)
            c.append(c_i)
        return K.concatenate(c, axis=1)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


def esi(embed_input1, embed_input2):
    ra1 = LSTM(200, activation='tanh')
    ra2 = LSTM(200, activation='tanh')
    ba1 = Bidirectional(ra1, merge_mode='concat')
    ba2 = Bidirectional(ra2, merge_mode='concat')
    da1 = Dense(200, activation='relu')
    da2 = Dense(1, activation='sigmoid')
    x = ba1(embed_input1)
    y = ba2(embed_input2)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = da1(z)
    z = Dropout(0.2)(z)
    return da2(z)
