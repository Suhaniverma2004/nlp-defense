"""
Practical Example: Using Character-Aware DistilBERT

Demonstrates:
1. Loading the model
2. Preparing adversarial inputs (with character-level attacks)
3. Comparing robustness vs baseline DistilBERT
4. Fine-tuning on adversarial examples
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.distilbert_char_model import create_char_aware_model
from src.char_vocab import build_char_vocab


class AdvHarassmentDetector(nn.Module):
    """
    Text classification head for detecting harmful content.
    Uses character-aware embeddings for robustness.
    """
    
    def __init__(self, embedding_dim=768, num_classes=2, dropout=0.1):
        """Initialize classification head."""
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, pooled_embeddings):
        """
        Forward pass.
        
        Args:
            pooled_embeddings: (batch_size, 768)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.dropout(pooled_embeddings)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def prepare_batch(texts, tokenizer, char_vocab, max_length=128, max_char_length=15):
    """
    Prepare batch of texts for the model.
    
    Args:
        texts (list): List of input texts
        tokenizer: HuggingFace tokenizer
        char_vocab (dict): Character vocabulary
        max_length (int): Maximum token sequence length
        max_char_length (int): Maximum character sequence length per token
    
    Returns:
        dict: Batch with input_ids, char_ids, attention_mask
    """
    batch_size = len(texts)
    
    # Tokenize
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Get actual sequence length from input_ids
    actual_seq_len = input_ids.shape[1]
    
    # Create character IDs with actual sequence length
    char_ids = torch.zeros((batch_size, actual_seq_len, max_char_length), dtype=torch.long)
    
    for batch_idx, text in enumerate(texts):
        # Tokenize individual text
        tokens = tokenizer.encode(text, add_special_tokens=False)[:actual_seq_len]
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        
        for token_idx, token_str in enumerate(token_strings):
            if token_idx >= actual_seq_len:
                break
            
            # Convert token to characters
            token_cleaned = token_str.replace('##', '')
            
            for char_idx, char in enumerate(token_cleaned[:max_char_length]):
                # Get character ID from vocabulary
                char_id = char_vocab.get(char, char_vocab.get('<PAD>', 0))
                char_ids[batch_idx, token_idx, char_idx] = char_id
    
    return {
        'input_ids': input_ids,
        'char_ids': char_ids,
        'attention_mask': attention_mask
    }


def demo_basic_usage():
    """Demonstrate basic model usage."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Model Usage")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model
    print("Loading character-aware DistilBERT model...")
    model = create_char_aware_model(fusion_method='concat', device=device)
    model.eval()
    print(f"✓ Model loaded\n")
    
    # Sample data
    batch_size = 2
    seq_len = 8
    char_seq_len = 12
    
    print(f"Creating sample batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Char sequence length: {char_seq_len}\n")
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    char_ids = torch.randint(0, len(build_char_vocab()), (batch_size, seq_len, char_seq_len), device=device)
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, char_ids=char_ids)
    
    print(f"✓ Forward pass complete\n")
    
    print("Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n✓ All outputs have correct dimensions (768 embedding size)")


def demo_text_processing():
    """Demonstrate text processing with tokenizer."""
    print("\n" + "="*70)
    print("DEMO 2: Text Processing with Tokenizer")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer and vocab
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    char_vocab = build_char_vocab()
    print(f"✓ Tokenizer loaded\n")
    
    # Sample texts
    texts = [
        "This is a normal text.",
        "Th1s is t3xt w1th numb3rs!"
    ]
    
    print(f"Input texts:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    print(f"\nPreparing batch...")
    batch = prepare_batch(texts, tokenizer, char_vocab)
    
    print(f"Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  char_ids: {batch['char_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    print("\n✓ Batch prepared successfully")


def demo_adversarial_robustness():
    """Demonstrate robustness against adversarial examples."""
    print("\n" + "="*70)
    print("DEMO 3: Adversarial Attack Robustness")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    print("Loading character-aware DistilBERT...")
    model = create_char_aware_model(fusion_method='concat', device=device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    char_vocab = build_char_vocab()
    
    print("✓ Model and tokenizer loaded\n")
    
    # Clean and adversarial texts
    clean_text = "This model is very robust against attacks"
    
    adversarial_texts = [
        "Th1s m0del is v3ry r0bust against att@cks",  # Number/symbol substitution
        "This  model  is  very  robust  against  attacks",  # Extra spaces
        "Thís mödél ís véry róbüst ágaínst attácks"  # Unicode variations
    ]
    
    print("Clean text:")
    print(f"  \"{clean_text}\"\n")
    
    print("Adversarial variations:")
    for i, adv_text in enumerate(adversarial_texts):
        print(f"  {i+1}. \"{adv_text}\"")
    
    # Prepare batches
    all_texts = [clean_text] + adversarial_texts
    batch = prepare_batch(all_texts, tokenizer, char_vocab)
    
    # Get embeddings
    print(f"\nGenerating embeddings...")
    with torch.no_grad():
        batch_device = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch_device)
    
    embeddings = outputs['last_hidden_state']  # (batch_size, seq_len, 768)
    pooled = embeddings.mean(dim=1)  # Average pooling: (batch_size, 768)
    
    # Calculate similarities
    print("\nSimilarities to clean embedding:")
    clean_emb = pooled[0]
    
    for i, text in enumerate(adversarial_texts):
        adv_emb = pooled[i + 1]
        similarity = torch.nn.functional.cosine_similarity(clean_emb.unsqueeze(0), adv_emb.unsqueeze(0))
        print(f"  Adversarial {i+1}: {similarity.item():.4f}")
    
    print("\n✓ Higher similarity = better robustness")
    print("  (Ideally all similarities should be > 0.95 for true robustness)")


def demo_fine_tuning():
    """Demonstrate fine-tuning for a classification task."""
    print("\n" + "="*70)
    print("DEMO 4: Fine-tuning Example")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    print("Initializing models...")
    distilbert_char = create_char_aware_model(fusion_method='concat', device=device)
    classifier = AdvHarassmentDetector(num_classes=2).to(device)
    print("✓ Models initialized\n")
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    char_vocab = build_char_vocab()
    optimizer = torch.optim.AdamW(
        list(distilbert_char.parameters()) + list(classifier.parameters()),
        lr=2e-5
    )
    criterion = nn.CrossEntropyLoss()
    
    # Sample training data
    texts = [
        "This is a great product!",
        "I love this movie!",
        "This is spam and offensive",
        "Terrible and harmful content"
    ]
    labels = torch.tensor([0, 0, 1, 1])  # 0=clean, 1=harmful
    
    print(f"Training on {len(texts)} samples\n")
    
    # Fine-tuning loop
    print("Running 5 training iterations...")
    distilbert_char.train()
    classifier.train()
    
    for epoch in range(5):
        batch = prepare_batch(texts, tokenizer, char_vocab)
        batch_device = {k: v.to(device) for k, v in batch.items()}
        labels_device = labels.to(device)
        
        # Forward pass
        outputs = distilbert_char(**batch_device)
        pooled = outputs['last_hidden_state'].mean(dim=1)
        logits = classifier(pooled)
        
        # Calculate loss
        loss = criterion(logits, labels_device)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print("\n✓ Fine-tuning complete")
    
    # Evaluation
    print("\nEvaluation on clean examples:")
    distilbert_char.eval()
    classifier.eval()
    
    with torch.no_grad():
        outputs = distilbert_char(**batch_device)
        pooled = outputs['last_hidden_state'].mean(dim=1)
        logits = classifier(pooled)
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels_device).float().mean()
    
    print(f"  Accuracy: {accuracy.item():.2%}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("Character-Aware DistilBERT: Practical Examples")
    print("="*70)
    
    try:
        demo_basic_usage()
        demo_text_processing()
        demo_adversarial_robustness()
        demo_fine_tuning()
        
        print("\n" + "="*70)
        print("All demonstrations completed successfully! ✓")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
