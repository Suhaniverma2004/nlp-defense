"""
Test script for Character Encoder

Tests the character encoder module:
- Generates random character IDs
- Passes them through the encoder
- Verifies output shape matches expected (batch_size, 768)
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from char_vocab import build_char_vocab
from char_encoder import CharEncoder


def test_char_encoder_output_shape():
    """Test that character encoder produces correct output shape."""
    print("=" * 60)
    print("Test: Character Encoder Output Shape")
    print("=" * 60)
    
    # Build vocabulary
    vocab = build_char_vocab()
    vocab_size = len(vocab)
    print(f"\n✓ Character vocabulary built with size: {vocab_size}")
    
    # Test parameters
    batch_size = 8
    char_seq_length = 20
    expected_output_dim = 768
    
    # Create character encoder
    encoder = CharEncoder(
        vocab_size=vocab_size,
        char_embed_dim=50,
        num_filters=100,
        filter_sizes=[2, 3, 4, 5],
        output_dim=expected_output_dim
    )
    print(f"✓ Character encoder created")
    
    # Generate random character IDs
    char_ids = torch.randint(0, vocab_size, (batch_size, char_seq_length))
    print(f"✓ Generated random character IDs with shape: {char_ids.shape}")
    
    # Forward pass
    output = encoder(char_ids)
    
    # Verify output shape
    expected_shape = torch.Size([batch_size, expected_output_dim])
    actual_shape = output.shape
    
    print(f"\nResults:")
    print(f"  Input shape:           {char_ids.shape}")
    print(f"  Output shape:          {actual_shape}")
    print(f"  Expected output shape: {expected_shape}")
    
    assert actual_shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {actual_shape}"
    print(f"\n✓ Output shape is correct!")
    
    # Check output properties
    print(f"\nOutput Properties:")
    print(f"  dtype:      {output.dtype}")
    print(f"  min value:  {output.min().item():.4f}")
    print(f"  max value:  {output.max().item():.4f}")
    print(f"  mean:       {output.mean().item():.4f}")
    print(f"  std:        {output.std().item():.4f}")
    
    return True


def test_batch_processing():
    """Test that encoder handles different batch sizes correctly."""
    print("\n" + "=" * 60)
    print("Test: Batch Processing")
    print("=" * 60)
    
    vocab = build_char_vocab()
    vocab_size = len(vocab)
    
    encoder = CharEncoder(vocab_size=vocab_size, output_dim=768)
    
    batch_sizes = [1, 4, 8, 16, 32]
    char_seq_length = 15
    
    for batch_size in batch_sizes:
        char_ids = torch.randint(0, vocab_size, (batch_size, char_seq_length))
        output = encoder(char_ids)
        
        expected_shape = torch.Size([batch_size, 768])
        assert output.shape == expected_shape, f"Batch size {batch_size} failed"
        print(f"✓ Batch size {batch_size:2d}: Output shape {output.shape}")
    
    return True


def test_variable_sequence_length():
    """Test that encoder handles variable sequence lengths."""
    print("\n" + "=" * 60)
    print("Test: Variable Sequence Lengths")
    print("=" * 60)
    
    vocab = build_char_vocab()
    vocab_size = len(vocab)
    
    encoder = CharEncoder(vocab_size=vocab_size, output_dim=768)
    
    batch_size = 4
    sequence_lengths = [5, 10, 15, 20, 30]
    
    for seq_len in sequence_lengths:
        char_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = encoder(char_ids)
        
        expected_shape = torch.Size([batch_size, 768])
        assert output.shape == expected_shape, f"Sequence length {seq_len} failed"
        print(f"✓ Sequence length {seq_len:2d}: Output shape {output.shape}")
    
    return True


def test_gradient_flow():
    """Test that gradients can flow through the encoder."""
    print("\n" + "=" * 60)
    print("Test: Gradient Flow")
    print("=" * 60)
    
    vocab = build_char_vocab()
    vocab_size = len(vocab)
    
    encoder = CharEncoder(vocab_size=vocab_size, output_dim=768)
    
    batch_size = 4
    char_seq_length = 10
    char_ids = torch.randint(0, vocab_size, (batch_size, char_seq_length))
    
    # Forward pass
    output = encoder(char_ids)
    
    # Compute a simple loss and backprop
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for param in encoder.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients found!"
    print(f"✓ Gradients successfully flow through encoder")
    
    return True


if __name__ == "__main__":
    try:
        test_char_encoder_output_shape()
        test_batch_processing()
        test_variable_sequence_length()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
