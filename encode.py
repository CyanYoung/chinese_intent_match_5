import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam

from build import bound_loss

from nn_arch import dnn_cache, cnn_cache, rnn_cache

from util import map_item


seq_len = 30

path_embed = 'feat/embed.pkl'
path_sent = 'feat/sent_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_sent, 'rb') as f:
    sents = pk.load(f)

funcs = {'dnn': dnn_cache,
         'cnn': cnn_cache,
         'rnn': rnn_cache}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}


def compile(name, embed_mat, seq_len, funcs):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,), dtype='int32')
    embed_input = embed(input)
    func = map_item(name, funcs)
    output = func(embed_input)
    model = Model(input, output)
    model.summary()
    model.compile(loss=bound_loss, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def cache(name, embed_mat, seq_len, sents, path_cache):
    model = compile(name, embed_mat, seq_len, funcs)
    model.load_weights(map_item(name, paths), by_name=True)
    encode_sents = model.predict(sents)
    with open(path_cache, 'wb') as f:
        f.write(encode_sents)


if __name__ == '__main__':
    path_cache = 'cache/dnn.pkl'
    cache('dnn', embed_mat, seq_len, sents, path_cache)
    # path_cache = 'cache/cnn.pkl'
    # cache('cnn', embed_mat, seq_len, sents, path_cache)
    # path_cache = 'cache/rnn.pkl'
    # cache('rnn', embed_mat, seq_len, sents, path_cache)
