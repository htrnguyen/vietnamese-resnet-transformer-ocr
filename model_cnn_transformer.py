import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class CNNEncoder(nn.Module):
    def __init__(self, d_model):
        super(CNNEncoder, self).__init__()
        base_model = models.resnet18(pretrained=True)
        modules = list(base_model.children())[:-2]  # Remove FC and AvgPool
        self.backbone = nn.Sequential(*modules)
        self.conv = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)         # (B, 512, H/32, W/32)
        feat = self.conv(feat)          # (B, d_model, H/32, W/32)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, H*W, d_model)
        return feat

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, max_len=36):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.output_layer(output)

class OCRModel(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super(OCRModel, self).__init__()
        self.encoder = CNNEncoder(d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

    def forward(self, images, tgt_seq):
        memory = self.encoder(images)                    # (B, Seq_len, d_model)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        output = self.decoder(tgt_seq, memory, tgt_mask=tgt_mask)
        return output
