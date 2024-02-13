import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class FEN(nn.Module):
    def __init__(self):
        super(FEN, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            SELayer(16),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            SELayer(32),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            SELayer(64),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2))
            
    def forward(self, x):
        x = self.conv_blocks(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Encoder(nn.Module):  #input [seq_len, batch_size, d_model]
    def __init__(self, d_model, nhead, dim_feedforward): 
        super(Encoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)   #output [seq_len, batch_size, d_model]
        return x

class TransformerHAR(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerHAR, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return x
    
class FeatureAttention(nn.Module):  #input [batch_size, seq_len, d_model]
    def __init__(self, d_model):
        super(FeatureAttention, self).__init__()
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        h = torch.tanh(self.W1(x))  # [batch_size, seq_len, d_model]
        a = F.softmax(self.W2(h), dim=1)  # [batch_size, seq_len, d_model]
        # a = self.dropout(a)
        C = torch.sum(a * x, dim=1)   # [batch_size, d_model]
        return C
    
class FLN(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes):
        super(FLN, self).__init__()

        self.temporal_stream = TransformerHAR(d_model, nhead, num_encoder_layers, dim_feedforward)

        self.feature_attention = FeatureAttention(d_model)
        
        self.decoder = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Temporal Stream
        x_temporal_atten = self.temporal_stream(x)   # [batch_size, win_len, d_model]
        x_temporal_atten = self.feature_attention(x_temporal_atten)  # [batch_size, d_model]
        # Classification
        x_temporal_atten = self.dropout(x_temporal_atten)    # [batch_size, d_model]
        x_temporal_atten = self.decoder(x_temporal_atten)    # [batch_size, num_classes]

        return x_temporal_atten

