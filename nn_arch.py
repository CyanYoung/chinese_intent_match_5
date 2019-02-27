import torch
import torch.nn as nn
import torch.nn.functional as F


class Esi(nn.Module):
    def __init__(self, embed_mat):
        super(Esi, self).__init__()
        self.encode1 = Encode1(embed_mat)
        self.encode2 = Encode2()
        self.lal = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200))
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, 1))

    @staticmethod
    def attend(x, y):
        d = torch.matmul(x, y.transpose(-2, -1))
        a = F.softmax(d, dim=-1)
        return torch.matmul(a, y)

    @staticmethod
    def merge1(x, x_):
        diff = torch.abs(x - x_)
        prod = x * x_
        return torch.cat([x, x_, diff, prod], dim=-1)

    def merge2(self, x, y):
        pass

    def forward(self, x, y):
        x, y = self.encode1(x), self.encode1(y)
        x_, y_ = self.attend(x, y), self.attend(y, x)
        x, y = self.merge1(x, x_), self.merge1(y, y_)
        x, y = self.encode2(x), self.encode2(y)
        z = self.merge2(x, y)
        z = self.lal(z)
        return self.dl(z)


class Encode1(nn.Module):
    def __init__(self, embed_mat):
        super(Encode1, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.ra = nn.LSTM(embed_len, 200, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embed(x)
        h, hc_n = self.ra(x)
        return h


class Encode2(nn.Module):
    def __init__(self):
        super(Encode2, self).__init__()
        self.ra = nn.LSTM(400 * 4, 200, batch_first=True, bidirectional=True)

    def forward(self, x):
        h, hc_n = self.ra(x)
        return h
