import math

import torch
import torch.nn as nn
import torchvision.models as models


class PositionalEncoding(nn.Module):
    """
    Positional encoding component for the Transformer model.
    Adds positional information to input embeddings.
    """

    def __init__(self, d_model, max_len=100):
        """
        Initialize positional encoding

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (not a parameter, but part of the module)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class CNNEncoder(nn.Module):
    """
    CNN-based encoder that extracts features from input images.
    Uses ResNet18 as the backbone feature extractor.
    """

    def __init__(self, d_model):
        """
        Initialize CNN encoder

        Args:
            d_model: Output feature dimension
        """
        super(CNNEncoder, self).__init__()
        # Use pretrained ResNet18 but remove classification layers
        base_model = models.resnet18(pretrained=True)
        modules = list(base_model.children())[:-2]  # Remove FC and AvgPool
        self.backbone = nn.Sequential(*modules)

        # Projection layer to d_model dimensions
        self.conv = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, x):
        """
        Extract features from input images

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Features of shape (batch_size, sequence_length, d_model)
        """
        # Extract features using ResNet backbone
        feat = self.backbone(x)  # (B, 512, H/32, W/32)

        # Project to required dimension
        feat = self.conv(feat)  # (B, d_model, H/32, W/32)

        # Reshape to sequence format
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, H*W, d_model)
        return feat


class TransformerDecoder(nn.Module):
    """
    Transformer decoder that processes features from the CNN encoder
    and generates output sequences.
    """

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, max_len=36):
        """
        Initialize transformer decoder

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            max_len: Maximum sequence length
        """
        super(TransformerDecoder, self).__init__()

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Process input sequences

        Args:
            tgt: Target sequence tensor
            memory: Memory tensor from encoder
            tgt_mask: Target mask for causal attention
            memory_mask: Memory mask

        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens and add positional encoding
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Pass through transformer decoder
        output = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        # Project to vocabulary size
        return self.output_layer(output)


class OCRModel(nn.Module):
    """
    Complete OCR model combining CNN encoder and Transformer decoder
    for optical character recognition tasks.
    """

    def __init__(self, vocab_size, d_model=256):
        """
        Initialize OCR model

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension used throughout the network
        """
        super(OCRModel, self).__init__()
        self.encoder = CNNEncoder(d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence to hide future tokens

        Args:
            sz: Sequence length

        Returns:
            Mask tensor
        """
        mask = torch.triu(torch.ones((sz, sz)) * float("-inf"), diagonal=1)
        return mask

    def forward(self, images, tgt_seq):
        """
        Forward pass of the OCR model

        Args:
            images: Input images of shape (batch_size, channels, height, width)
            tgt_seq: Target sequence tensor

        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        # Extract features from images
        memory = self.encoder(images)

        # Generate causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq.size(1)).to(
            tgt_seq.device
        )

        # Decode sequences
        output = self.decoder(tgt_seq, memory, tgt_mask=tgt_mask)

        return output
