"""
Character Encoder Module

CNN-based character encoder that produces fixed-length vector representations
for sequences of characters. Outputs 768-dimensional vectors matching DistilBERT
hidden size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.char_vocab import build_char_vocab


class CharEncoder(nn.Module):
    """
    CNN-based Character Encoder
    
    Encodes character sequences into fixed-length vectors suitable for
    integration with DistilBERT embeddings.
    
    Architecture:
        char_ids → char_embedding → conv_layers → max_pool → FC → output(768)
    
    Args:
        vocab_size (int): Size of character vocabulary
        char_embed_dim (int): Dimension of character embeddings (default: 50)
        num_filters (int): Number of filters per filter size (default: 100)
        filter_sizes (list): Filter sizes for conv layers (default: [2, 3, 4, 5])
        output_dim (int): Output embedding dimension (default: 768 for DistilBERT)
        padding_idx (int): Padding token index (default: 0)
    """
    
    def __init__(
        self,
        vocab_size,
        char_embed_dim=50,
        num_filters=100,
        filter_sizes=None,
        output_dim=768,
        padding_idx=0
    ):
        """Initialize the character encoder."""
        super(CharEncoder, self).__init__()
        
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]
        
        self.vocab_size = vocab_size
        self.char_embed_dim = char_embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.output_dim = output_dim
        self.padding_idx = padding_idx
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=char_embed_dim,
            padding_idx=padding_idx
        )
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_embed_dim,
                out_channels=num_filters,
                kernel_size=filter_size,
                padding=filter_size - 1  # "same" padding
            )
            for filter_size in filter_sizes
        ])
        
        # Total number of features after conv and max pooling
        total_conv_features = num_filters * len(filter_sizes)
        
        # Fully connected layers to project to output_dim
        self.fc1 = nn.Linear(total_conv_features, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
    
    def forward(self, char_ids):
        """
        Forward pass through character encoder.
        
        Args:
            char_ids (torch.Tensor): Character indices
                Shape: (batch_size, char_sequence_length)
        
        Returns:
            torch.Tensor: Fixed-length char encodings
                Shape: (batch_size, output_dim)
        """
        # Embed characters: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded = self.char_embedding(char_ids)
        
        # Transpose for Conv1d: (batch_size, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutional layers with max pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution: (batch_size, num_filters, seq_len)
            conv_out = self.relu(conv(embedded))
            
            # Max pool over sequence dimension
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            
            # Remove sequence dimension: (batch_size, num_filters)
            pooled = pooled.squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs: (batch_size, num_filters * len(filter_sizes))
        merged = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        merged = self.dropout(merged)
        
        # Pass through fully connected layers
        hidden = self.relu(self.fc1(merged))
        hidden = self.dropout(hidden)
        
        # Output layer: (batch_size, output_dim)
        output = self.fc2(hidden)
        
        return output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    # Example usage
    vocab = build_char_vocab()
    vocab_size = len(vocab)
    
    print(f"Character vocabulary size: {vocab_size}")
    
    # Create character encoder
    encoder = CharEncoder(
        vocab_size=vocab_size,
        char_embed_dim=50,
        num_filters=100,
        filter_sizes=[2, 3, 4, 5],
        output_dim=768
    )
    
    # Create random character IDs
    batch_size = 4
    char_seq_length = 20  # Average token length consideration
    char_ids = torch.randint(0, vocab_size, (batch_size, char_seq_length))
    
    # Forward pass
    output = encoder(char_ids)
    
    print(f"\nInput shape (batch_size, char_sequence_length): {char_ids.shape}")
    print(f"Output shape (batch_size, output_dim): {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"\nExpected output shape: torch.Size([{batch_size}, 768])")
    print(f"Actual output shape: {output.shape}")
    print(f"Match: {output.shape == torch.Size([batch_size, 768])}")
