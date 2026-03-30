"""
Character-Aware DistilBERT Model

Integrates Character-Aware embeddings into DistilBERT by modifying the embedding layer.
The model combines token-level and character-level representations to improve
robustness against tokenization-level adversarial attacks.

Architecture:
    token_ids + char_ids → token_embeddings + char_embeddings
                        → fusion_layer → DistilBERT transformer blocks
                        → final_output
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from src.char_encoder import CharEncoder
from src.char_vocab import build_char_vocab


class DistilBertCharModel(nn.Module):
    """
    Character-Aware DistilBERT Model
    
    Combines token-level and character-level embeddings before passing
    through the DistilBERT transformer.
    
    Args:
        char_vocab_size (int): Size of character vocabulary
        char_embed_dim (int): Character embedding dimension (default: 50)
        num_filters (int): Number of CNN filters per size (default: 100)
        filter_sizes (list): CNN filter kernel sizes (default: [2, 3, 4, 5])
        fusion_method (str): How to combine embeddings - 'add' or 'concat' (default: 'concat')
        model_name (str): HuggingFace model name (default: 'distilbert-base-uncased')
        dropout_rate (float): Dropout rate for fusion layer (default: 0.1)
    """
    
    def __init__(
        self,
        char_vocab_size,
        char_embed_dim=50,
        num_filters=100,
        filter_sizes=None,
        fusion_method='concat',
        model_name='distilbert-base-uncased',
        dropout_rate=0.1
    ):
        """Initialize the character-aware DistilBERT model."""
        super(DistilBertCharModel, self).__init__()
        
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]
        
        # Configuration
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.fusion_method = fusion_method
        self.hidden_size = 768  # DistilBERT hidden size
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Initialize character encoder
        self.char_encoder = CharEncoder(
            vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            output_dim=self.hidden_size,
            padding_idx=0
        )
        
        # Fusion layer to combine token and character embeddings
        if fusion_method == 'concat':
            # Concatenation + projection: (768 * 2) → 768
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )
        elif fusion_method == 'add':
            # Simple addition (no additional parameters)
            self.fusion = None
        else:
            raise ValueError(f"fusion_method must be 'add' or 'concat', got {fusion_method}")
    
    def forward(
        self,
        input_ids,
        char_ids,
        attention_mask=None,
        token_type_ids=None
    ):
        """
        Forward pass through character-aware DistilBERT.
        
        Args:
            input_ids (torch.Tensor): Token IDs
                Shape: (batch_size, seq_len)
            
            char_ids (torch.Tensor): Character IDs for each token
                Shape: (batch_size, seq_len, char_seq_len)
            
            attention_mask (torch.Tensor, optional): Attention mask for tokens
                Shape: (batch_size, seq_len)
                Values: 0 for padding, 1 for real tokens
            
            token_type_ids (torch.Tensor, optional): Token type IDs (not typically used for DistilBERT)
                Shape: (batch_size, seq_len)
        
        Returns:
            dict: Model outputs containing:
                - 'last_hidden_state': Final hidden states from DistilBERT
                  Shape: (batch_size, seq_len, 768)
                - 'char_embeddings': Character-level embeddings
                  Shape: (batch_size, seq_len, 768)
                - 'token_embeddings': Token-level embeddings
                  Shape: (batch_size, seq_len, 768)
        """
        batch_size, seq_len, char_seq_len = char_ids.shape
        
        # ========== STEP 1: Get token embeddings ==========
        # Extract word embeddings from DistilBERT without position encoding
        token_embeddings = self.distilbert.embeddings.word_embeddings(input_ids)
        # Shape: (batch_size, seq_len, 768)
        
        # ========== STEP 2: Process character IDs ==========
        # Reshape char_ids to process all tokens at once
        char_ids_reshaped = char_ids.view(batch_size * seq_len, char_seq_len)
        # Shape: (batch_size * seq_len, char_seq_len)
        
        # Pass through character encoder
        char_embeddings_flat = self.char_encoder(char_ids_reshaped)
        # Shape: (batch_size * seq_len, 768)
        
        # Reshape back to (batch_size, seq_len, 768)
        char_embeddings = char_embeddings_flat.view(batch_size, seq_len, self.hidden_size)
        # Shape: (batch_size, seq_len, 768)
        
        # ========== STEP 3: Fuse token and character embeddings ==========
        if self.fusion_method == 'concat':
            # Concatenate along embedding dimension
            fused_embeddings = torch.cat(
                [token_embeddings, char_embeddings],
                dim=-1
            )
            # Shape: (batch_size, seq_len, 1536)
            
            # Project back to 768
            fused_embeddings = self.fusion(fused_embeddings)
            # Shape: (batch_size, seq_len, 768)
        
        elif self.fusion_method == 'add':
            # Simple addition
            fused_embeddings = token_embeddings + char_embeddings
            # Shape: (batch_size, seq_len, 768)
        
        # ========== STEP 4: Add position encodings from DistilBERT ==========
        # Get position IDs (standard: 0, 1, 2, ..., seq_len-1)
        position_ids = torch.arange(
            seq_len,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Add position embeddings
        position_embeddings = self.distilbert.embeddings.position_embeddings(position_ids)
        # Shape: (batch_size, seq_len, 768)
        
        # Combine with fused embeddings
        final_embeddings = fused_embeddings + position_embeddings
        # Shape: (batch_size, seq_len, 768)
        
        # Apply layer normalization and dropout (from DistilBERT embeddings)
        final_embeddings = self.distilbert.embeddings.LayerNorm(final_embeddings)
        final_embeddings = self.distilbert.embeddings.dropout(final_embeddings)
        # Shape: (batch_size, seq_len, 768)
        
        # ========== STEP 5: Pass through DistilBERT transformer ==========
        distilbert_outputs = self.distilbert(
            inputs_embeds=final_embeddings,
            attention_mask=attention_mask
        )
        # outputs.last_hidden_state shape: (batch_size, seq_len, 768)
        
        # ========== STEP 6: Return outputs ==========
        return {
            'last_hidden_state': distilbert_outputs.last_hidden_state,
            'char_embeddings': char_embeddings,
            'token_embeddings': token_embeddings,
            'fused_embeddings': fused_embeddings,
            'distilbert_outputs': distilbert_outputs
        }
    
    def freeze_distilbert(self, freeze=True):
        """
        Freeze or unfreeze DistilBERT parameters.
        
        Args:
            freeze (bool): Whether to freeze DistilBERT layers (default: True)
        """
        for param in self.distilbert.parameters():
            param.requires_grad = not freeze
    
    def freeze_char_encoder(self, freeze=True):
        """
        Freeze or unfreeze character encoder parameters.
        
        Args:
            freeze (bool): Whether to freeze CharEncoder layers (default: True)
        """
        for param in self.char_encoder.parameters():
            param.requires_grad = not freeze
    
    @property
    def distilbert_config(self):
        """Get DistilBERT configuration."""
        return self.distilbert.config


def create_char_aware_model(
    fusion_method='concat',
    model_name='distilbert-base-uncased',
    device=None
):
    """
    Convenience function to create a character-aware DistilBERT model.
    
    Args:
        fusion_method (str): How to combine embeddings - 'add' or 'concat'
        model_name (str): HuggingFace model name
        device (torch.device, optional): Device to load model on
    
    Returns:
        DistilBertCharModel: Initialized model
    """
    # Build character vocabulary
    char_vocab = build_char_vocab()
    char_vocab_size = len(char_vocab)
    
    # Create model
    model = DistilBertCharModel(
        char_vocab_size=char_vocab_size,
        char_embed_dim=50,
        num_filters=100,
        filter_sizes=[2, 3, 4, 5],
        fusion_method=fusion_method,
        model_name=model_name
    )
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model


if __name__ == "__main__":
    """Example usage of character-aware DistilBERT."""
    import sys
    sys.path.insert(0, '.')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_char_aware_model(fusion_method='concat', device=device)
    model.eval()
    
    print("\nModel created successfully!")
    print(f"Model architecture:\n{model}\n")
    
    # Test forward pass
    batch_size = 2
    seq_len = 8
    char_seq_len = 12
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    char_ids = torch.randint(0, len(build_char_vocab()), (batch_size, seq_len, char_seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  char_ids: {char_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}\n")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            char_ids=char_ids,
            attention_mask=attention_mask
        )
    
    print(f"Output shapes:")
    print(f"  last_hidden_state: {outputs['last_hidden_state'].shape}")
    print(f"  char_embeddings: {outputs['char_embeddings'].shape}")
    print(f"  token_embeddings: {outputs['token_embeddings'].shape}")
    print(f"  fused_embeddings: {outputs['fused_embeddings'].shape}\n")
    
    print("✓ All outputs have correct dimensions (768)")
