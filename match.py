import pickle as pk

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from util import load_word_re, load_pair, map_item


seq_len = 30

path_stop_word = 'dict/stop_word.txt'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

paths = {'esi': 'model/rnn_esi.h5'}

models = {'esi': load_model(map_item('esi', paths))}


def predict(text1, text2, name):
    text1, text2 = clean(text1), clean(text2)
    seq1 = word2ind.texts_to_sequences([text1])[0]
    seq2 = word2ind.texts_to_sequences([text2])[0]
    pad_seq1 = pad_sequences([seq1], maxlen=seq_len)
    pad_seq2 = pad_sequences([seq2], maxlen=seq_len)
    model = map_item(name, models)
    prob = model.predict([pad_seq1, pad_seq2])[0][0]
    return '{:.3f}'.format(prob)


if __name__ == '__main__':
    while True:
        text1, text2 = input('text1: '), input('text2: ')
        print('esi: %s' % predict(text1, text2, 'esi'))
