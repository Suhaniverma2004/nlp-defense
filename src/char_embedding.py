"""
Character Embedding Layer

PyTorch embedding layer for character-level representations.
"""

import torch
import torch.nn as nn


class CharEmbedding(nn.Module):
    """
    Character Embedding Layer
    
    Converts character indices to dense embeddings.
    
    Args:
        vocab_size (int): Size of character vocabulary
        embedding_dim (int): Dimension of character embeddings (default: 50)
        padding_idx (int): Index of padding token (default: 0)
    """
    
    def __init__(self, vocab_size, embedding_dim=50, padding_idx=0):
        """Initialize the embedding layer."""
        super(CharEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Create embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
    
    def forward(self, char_ids):
        """
        Forward pass through embedding layer.
        
        Args:
            char_ids (torch.Tensor): Character indices
                Shape: (batch_size, char_sequence_length)
        
        Returns:
            torch.Tensor: Character embeddings
                Shape: (batch_size, char_sequence_length, embedding_dim)
        """
        return self.embedding(char_ids)


if __name__ == "__main__":
    # Example usage
    vocab_size = 100  # Size of character vocabulary
    embedding_dim = 50
    batch_size = 4
    seq_length = 10
    
    # Create embedding layer
    char_emb = CharEmbedding(vocab_size, embedding_dim)
    
    # Create random character IDs
    char_ids = torch.randint(1, vocab_size, (batch_size, seq_length))
    
    # Pass through embedding layer
    embeddings = char_emb(char_ids)
    
    print(f"Input shape: {char_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_length}, {embedding_dim})")
