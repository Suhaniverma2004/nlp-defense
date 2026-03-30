"""
Test suite for Character-Aware DistilBERT Model

Tests:
- Model initialization
- Forward pass with correct input/output shapes
- Different fusion methods
- Attention mask handling
- Gradient flow
- Device compatibility
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import unittest
from src.distilbert_char_model import DistilBertCharModel, create_char_aware_model
from src.char_vocab import build_char_vocab


class TestDistilBertCharModel(unittest.TestCase):
    """Test cases for DistilBertCharModel."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.char_vocab = build_char_vocab()
        cls.char_vocab_size = len(cls.char_vocab)
        
    def setUp(self):
        """Set up for each test."""
        self.batch_size = 4
        self.seq_len = 10
        self.char_seq_len = 15
        self.hidden_size = 768
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size,
            char_embed_dim=50,
            num_filters=100,
            filter_sizes=[2, 3, 4, 5],
            fusion_method='concat'
        )
        
        self.assertIsNotNone(model.distilbert)
        self.assertIsNotNone(model.char_encoder)
        self.assertIsNotNone(model.fusion)
        self.assertEqual(model.hidden_size, 768)
    
    def test_model_initialization_add_fusion(self):
        """Test model initialization with add fusion method."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size,
            fusion_method='add'
        )
        
        self.assertIsNone(model.fusion)
        self.assertEqual(model.fusion_method, 'add')
    
    def test_invalid_fusion_method(self):
        """Test that invalid fusion method raises ValueError."""
        with self.assertRaises(ValueError):
            DistilBertCharModel(
                char_vocab_size=self.char_vocab_size,
                fusion_method='invalid'
            )
    
    def test_forward_pass_concat_fusion(self):
        """Test forward pass with concatenation fusion."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size,
            fusion_method='concat'
        ).to(self.device)
        model.eval()
        
        input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_len), device=self.device)
        char_ids = torch.randint(0, self.char_vocab_size, (self.batch_size, self.seq_len, self.char_seq_len), device=self.device)
        attention_mask = torch.ones((self.batch_size, self.seq_len), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                char_ids=char_ids,
                attention_mask=attention_mask
            )
        
        # Check output shapes
        self.assertEqual(outputs['last_hidden_state'].shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(outputs['char_embeddings'].shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(outputs['token_embeddings'].shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(outputs['fused_embeddings'].shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_forward_pass_add_fusion(self):
        """Test forward pass with addition fusion."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size,
            fusion_method='add'
        ).to(self.device)
        model.eval()
        
        input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_len), device=self.device)
        char_ids = torch.randint(0, self.char_vocab_size, (self.batch_size, self.seq_len, self.char_seq_len), device=self.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                char_ids=char_ids
            )
        
        self.assertEqual(outputs['last_hidden_state'].shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_attention_mask_handling(self):
        """Test that attention mask is properly applied."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size,
            fusion_method='concat'
        ).to(self.device)
        model.eval()
        
        input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_len), device=self.device)
        char_ids = torch.randint(0, self.char_vocab_size, (self.batch_size, self.seq_len, self.char_seq_len), device=self.device)
        
        # Attention mask: mask out last 2 tokens
        attention_mask = torch.ones((self.batch_size, self.seq_len), dtype=torch.long, device=self.device)
        attention_mask[:, -2:] = 0
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                char_ids=char_ids,
                attention_mask=attention_mask
            )
        
        self.assertEqual(outputs['last_hidden_state'].shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_batch_size_one(self):
        """Test model with batch size 1."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size
        ).to(self.device)
        model.eval()
        
        input_ids = torch.randint(0, 30522, (1, self.seq_len), device=self.device)
        char_ids = torch.randint(0, self.char_vocab_size, (1, self.seq_len, self.char_seq_len), device=self.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, char_ids=char_ids)
        
        self.assertEqual(outputs['last_hidden_state'].shape, (1, self.seq_len, self.hidden_size))
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size
        ).to(self.device)
        model.eval()
        
        seq_lengths = [5, 10, 20, 32]
        
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, 30522, (self.batch_size, seq_len), device=self.device)
            char_ids = torch.randint(0, self.char_vocab_size, (self.batch_size, seq_len, self.char_seq_len), device=self.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, char_ids=char_ids)
            
            self.assertEqual(outputs['last_hidden_state'].shape, (self.batch_size, seq_len, self.hidden_size))
    
    def test_create_char_aware_model_function(self):
        """Test the create_char_aware_model convenience function."""
        model = create_char_aware_model(fusion_method='concat', device=self.device)
        
        self.assertIsInstance(model, DistilBertCharModel)
        self.assertEqual(model.fusion_method, 'concat')
        
        # Test forward pass
        input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_len), device=self.device)
        char_ids = torch.randint(0, len(build_char_vocab()), (self.batch_size, self.seq_len, self.char_seq_len), device=self.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, char_ids=char_ids)
        
        self.assertEqual(outputs['last_hidden_state'].shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_freeze_distilbert(self):
        """Test freezing DistilBERT parameters."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size
        )
        
        model.freeze_distilbert(freeze=True)
        
        # Check that DistilBERT params are frozen
        for param in model.distilbert.parameters():
            self.assertFalse(param.requires_grad)
        
        # Check that CharEncoder params are not frozen
        for param in model.char_encoder.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_freeze_char_encoder(self):
        """Test freezing CharEncoder parameters."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size
        )
        
        model.freeze_char_encoder(freeze=True)
        
        # Check that CharEncoder params are frozen
        for param in model.char_encoder.parameters():
            self.assertFalse(param.requires_grad)
        
        # Check that DistilBERT params are not frozen
        for param in model.distilbert.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size,
            fusion_method='concat'
        ).to(self.device)
        model.train()
        
        input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_len), device=self.device)
        char_ids = torch.randint(0, self.char_vocab_size, (self.batch_size, self.seq_len, self.char_seq_len), device=self.device)
        
        outputs = model(input_ids=input_ids, char_ids=char_ids)
        loss = outputs['last_hidden_state'].sum()
        loss.backward()
        
        # Check that gradients exist in char_encoder
        for param in model.char_encoder.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
        
        # Check that gradients exist in fusion layer
        for param in model.fusion.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_output_dtype(self):
        """Test that output dtype matches input dtype."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size
        ).to(self.device)
        model.eval()
        
        input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_len), device=self.device, dtype=torch.long)
        char_ids = torch.randint(0, self.char_vocab_size, (self.batch_size, self.seq_len, self.char_seq_len), device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, char_ids=char_ids)
        
        # Check that output is float (default dtype for embeddings)
        self.assertTrue(outputs['last_hidden_state'].dtype in [torch.float32, torch.float64])
    
    def test_model_config_property(self):
        """Test that distilbert_config property works."""
        model = DistilBertCharModel(
            char_vocab_size=self.char_vocab_size
        )
        
        config = model.distilbert_config
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.model_type, 'distilbert')


class TestDistilBertCharModelIntegration(unittest.TestCase):
    """Integration tests for the model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_full_pipeline(self):
        """Test a complete pipeline with model creation and inference."""
        # Create model
        model = create_char_aware_model(
            fusion_method='concat',
            device=self.device
        )
        model.eval()
        
        # Create inputs
        batch_size = 2
        seq_len = 8
        char_seq_len = 10
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=self.device)
        char_ids = torch.randint(0, len(build_char_vocab()), (batch_size, seq_len, char_seq_len), device=self.device)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                char_ids=char_ids,
                attention_mask=attention_mask
            )
        
        # Verify outputs
        self.assertIn('last_hidden_state', outputs)
        self.assertIn('char_embeddings', outputs)
        self.assertIn('token_embeddings', outputs)
        self.assertIn('fused_embeddings', outputs)
        
        # Verify shapes
        last_hidden = outputs['last_hidden_state']
        self.assertEqual(last_hidden.shape, (batch_size, seq_len, 768))
        self.assertGreater(last_hidden.abs().max().item(), 0)  # Not all zeros
    
    def test_deterministic_inference(self):
        """Test that inference is deterministic with same seed."""
        torch.manual_seed(42)
        model1 = create_char_aware_model(device=self.device)
        model1.eval()
        
        input_ids = torch.randint(0, 30522, (2, 8), device=self.device)
        char_ids = torch.randint(0, len(build_char_vocab()), (2, 8, 10), device=self.device)
        
        with torch.no_grad():
            outputs1 = model1(input_ids=input_ids, char_ids=char_ids)
        
        torch.manual_seed(42)
        model2 = create_char_aware_model(device=self.device)
        model2.eval()
        
        with torch.no_grad():
            outputs2 = model2(input_ids=input_ids, char_ids=char_ids)
        
        # Check that outputs are the same (with small tolerance for floating point)
        torch.testing.assert_close(
            outputs1['last_hidden_state'],
            outputs2['last_hidden_state'],
            rtol=1e-4,
            atol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
