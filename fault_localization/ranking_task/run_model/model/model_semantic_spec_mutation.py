from torch import nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.linear_layer = nn.Linear(10, 60)

    def forward(self, x):
        residual = x

        #out = self.conv1(x)
        #out = self.relu(out)
        #out = self.conv2(out)

        #out += residual  # 跳跃连接
        out = self.linear_layer(x)

        out = self.relu(out)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector, attention_weights

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.gru = nn.GRU(10, 60, 2, batch_first=True, dropout=0.4)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        self.attention = Attention(60)
        self.fc = nn.Linear(60, 4)
        nn.init.xavier_uniform_(self.fc.weight)
        self.mlp_semantic = nn.Linear(11, 3)
        nn.init.xavier_uniform_(self.mlp_semantic.weight)
        nn.init.zeros_(self.mlp_semantic.bias)

        self.mlp_all_features = nn.Linear(60, 8)
        nn.init.xavier_uniform_(self.mlp_all_features.weight)
        self.residual_block1 = ResidualBlock(60,60)

        self.output_layer = nn.Linear(8, 2)
        #nn.init.xavier_uniform_(self.output_layer.weight)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.4)



    def forward(self, inputs):
        spectrum = inputs[:, 0:3]
        mutation = inputs[:, 3:7]
        semantic = inputs[:, 7:]

        semantic = self.dropout(self.activation(self.mlp_semantic(semantic)))
        all_features = torch.cat([spectrum, mutation, semantic], dim=-1)
        all_features = all_features.view(len(all_features), 1, -1)
        #all_features_orig = all_features.clone()
        h0 = torch.zeros(2, all_features.size(0), 60)
        h0 = h0.to('cuda:0')
        out, hidden = self.gru(all_features, h0)

        #all_features_orig = self.residual_block1(all_features_orig)
        #out = self.residual_block2(out)
        all_features, attention_weights = self.attention(hidden[-1], out)
        #all_features_orig = all_features_orig.squeeze(dim=1)
        #print(all_features.shape)
        #print(all_features_orig.shape)
        #all_features += all_features_orig  # 将原始输入与注意力层输出相加
        all_features = self.dropout(self.activation(self.mlp_all_features(all_features)))
        #all_features = self.batch_norm(all_features)

        out = self.output_layer(all_features)
        return out
