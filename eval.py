import pickle as pk

import numpy as np

from keras.models import load_model

from sklearn.metrics import accuracy_score

from util import flat_read, map_item


path_test = 'data/test.csv'
path_train_sent = 'feat/sent_train.pkl'
path_train_label = 'feat/label_train.pkl'
path_test_sent = 'feat/sent_test.pkl'
path_test_label = 'feat/label_test.pkl'
texts = flat_read(path_test, 'text')
with open(path_train_sent, 'rb') as f:
    train_sents = pk.load(f)
with open(path_train_label, 'rb') as f:
    train_labels = pk.load(f)
with open(path_test_sent, 'rb') as f:
    test_sents = pk.load(f)
with open(path_test_label, 'rb') as f:
    test_labels = pk.load(f)

path_test_pair = 'data/test_pair.csv'
path_pair = 'feat/pair_train.pkl'
path_flag = 'feat/flag_train.pkl'
text1s = flat_read(path_test_pair, 'text1')
text2s = flat_read(path_test_pair, 'text2')
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)
with open(path_flag, 'rb') as f:
    flags = pk.load(f)

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}

models = {'dnn': load_model(map_item('dnn', paths)),
          'cnn': load_model(map_item('cnn', paths)),
          'rnn': load_model(map_item('rnn', paths))}


def test_pair(name, pairs, flags, thre):
    model = map_item(name, models)
    sent1s, sent2s = pairs
    dists = model.predict([sent1s, sent2s])
    dists = np.reshape(dists, (1, -1))[0]
    preds = dists > thre
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(flags, preds)))
    for flag, dist, text1, text2, pred in zip(flags, dists, text1s, text2s, preds):
        if flag != pred:
            print('{} {:.3f} {} | {}'.format(flag, dist, text1, text2))


def test(name, test_sents, test_labels, train_sents, train_labels):
    model = map_item(name, models)
    preds = list()
    for test_sent in test_sents:
        test_mat = np.repeat([test_sent], len(train_sents), axis=0)
        dists = model.predict([test_mat, train_sents])
        dists = np.reshape(dists, (1, -1))[0]
        min_ind = np.argmin(dists)
        preds.append(train_labels[min_ind])
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(test_labels, preds)))
    for text, label, pred in zip(texts, test_labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    test_pair('dnn', pairs, flags, thre=0.5)
    test_pair('cnn', pairs, flags, thre=0.5)
    test_pair('rnn', pairs, flags, thre=0.5)
    test('dnn', test_sents, test_labels, train_sents, train_labels)
    test('cnn', test_sents, test_labels, train_sents, train_labels)
    test('rnn', test_sents, test_labels, train_sents, train_labels)
