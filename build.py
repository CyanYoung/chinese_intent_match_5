import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import keras.backend as K

from nn_arch import dnn_build, cnn_build, rnn_build

from util import map_item


batch_size = 32

path_embed = 'feat/embed.pkl'
path_pair = 'feat/pair_train.pkl'
path_flag = 'feat/flag_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)
with open(path_flag, 'rb') as f:
    flags = pk.load(f)

funcs = {'dnn': dnn_build,
         'cnn': cnn_build,
         'rnn': rnn_build}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}


def bound_loss(dist, flag):
    return flag * K.maximum(0.0, flag - dist) + (1.0 - flag) * dist


def compile(name, embed_mat, seq_len, funcs):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input1 = Input(shape=(seq_len,), dtype='int32')
    input2 = Input(shape=(seq_len,), dtype='int32')
    embed_input1 = embed(input1)
    embed_input2 = embed(input2)
    func = map_item(name, funcs)
    output = func(embed_input1, embed_input2)
    model = Model([input1, input2], output)
    model.summary()
    model.compile(loss=bound_loss, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def fit(name, epoch, embed_mat, pairs, flags):
    sent1s, sent2s = pairs
    seq_len = len(sent1s[0])
    model = compile(name, embed_mat, seq_len, funcs)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit([sent1s, sent2s], flags, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    fit('dnn', 30, embed_mat, pairs, flags)
    # fit('cnn', 30, embed_mat, pairs, flags)
    # fit('rnn', 20, embed_mat, pairs, flags)
