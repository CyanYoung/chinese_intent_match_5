import torch
import torch.nn as nn


class Esi(nn.Module):
    def __init__(self, embed_mat):
        super(Esi, self).__init__()
        self.encode = RnnEncode(embed_mat)

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        return self.match(x, y)


class RnnEncode(nn.Module):
    def __init__(self, embed_mat):
        super(RnnEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len)
        self.ra = nn.LSTM(embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        h, hc_n = self.ra(x)
        return h[:, -1, :]


class Match(nn.Module):
    def __init__(self):
        super(Match, self).__init__()
        self.la = nn.Sequential(nn.Linear(800, 200),
                                nn.ReLU())
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, 1))

    def forward(self, x, y):
        diff = torch.abs(x - y)
        prod = x * y
        z = torch.cat([x, y, diff, prod], dim=1)
        z = self.la(z)
        return self.dl(z)
