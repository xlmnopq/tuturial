"""IMDB 二分类模型。两种模型共用词表、分词器、数据划分。"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                               bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 2))

    def forward(self, tokens, lengths):
        embeddings = self.embedding(tokens)
        # 用真实长度打包，避免最后的 PAD 污染句向量。
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        # 一层双向 LSTM：hidden[-2]、hidden[-1] 是两方向的最终状态。
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(features)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_channels=128,
                 kernels=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.kernels = tuple(kernels)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_channels, k) for k in self.kernels])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(num_channels * len(self.kernels), 2))

    def forward(self, tokens, lengths):
        # [batch, length, embedding] -> Conv1d 所需的 [batch, embedding, length]
        features = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for conv, kernel in zip(self.convs, self.kernels):
            values = torch.relu(conv(features))
            # 排除完全落在句子 PAD 区间的窗口，减少批内填充长度的影响。
            # 短于卷积核的句子保留一个用 PAD 补齐的窗口。
            valid = (lengths - kernel + 1).clamp(min=1)
            positions = torch.arange(values.size(2), device=values.device)
            mask = positions.unsqueeze(0) >= valid.unsqueeze(1)
            values = values.masked_fill(mask.unsqueeze(1), float("-inf"))
            pooled.append(values.max(dim=2).values)
        return self.classifier(torch.cat(pooled, dim=1))


def build_text_model(name, vocab_size, **kwargs):
    if name == "bilstm":
        return BiLSTM(vocab_size, **kwargs)
    if name == "textcnn":
        return TextCNN(vocab_size, **kwargs)
    raise ValueError("不支持的文本模型: " + name)
