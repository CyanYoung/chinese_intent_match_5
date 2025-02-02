import pickle as pk

import torch

from sklearn.metrics import accuracy_score, f1_score

from build import tensorize

from match import models

from util import map_item


device = torch.device('cpu')

path_pair = 'feat/pair_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, pairs, labels, thre):
    sent1s, sent2s = pairs
    sent1s, sent2s, labels = tensorize([sent1s, sent2s, labels], device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sent1s, sent2s))
    probs = torch.squeeze(probs, dim=-1)
    preds = probs > thre
    f1 = f1_score(labels, preds)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(labels, preds)))


if __name__ == '__main__':
    test('esi', pairs, labels, thre=0.2)
