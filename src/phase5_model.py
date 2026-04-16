import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from src.char_vocab import build_char_vocab

# 🔥 NEW IMPORTS
from src.sanitization import sanitize_text
from src.composite_defense import composite_clean, is_composite

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
char_vocab = build_char_vocab()

# ---------------- CHAR CNN ----------------
class CharCNN(nn.Module):
    def __init__(self, vocab_size, char_emb_dim=50, out_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(char_emb_dim, 128, k) for k in [2, 3, 4, 5]
        ])

        self.fc = nn.Linear(128 * 4, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        B, S, C, E = x.shape

        x = x.view(B * S, C, E).permute(0, 2, 1)

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max(c, dim=2)[0]
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=1)
        x = self.fc(x)

        return x.view(B, S, -1)


# ---------------- MAIN MODEL ----------------
class CharAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.char_encoder = CharCNN(len(char_vocab))
        self.fusion = nn.Linear(768 * 2, 768)

    def forward(self, input_ids, char_ids, attention_mask):
        token_embeds = self.bert.embeddings(input_ids)
        char_embeds = self.char_encoder(char_ids)

        combined = torch.cat([token_embeds, char_embeds], dim=-1)
        fused = self.fusion(combined)

        outputs = self.bert(inputs_embeds=fused, attention_mask=attention_mask)
        return outputs.last_hidden_state


# ---------------- 🔥 NEW: TEXT PROCESSING PIPELINE ----------------
def process_text(text):
    """
    Phase II + Composite Defense Layer
    """
    if is_composite(text):
        text = composite_clean(text)
    else:
        text = sanitize_text(text)

    return text


# ---------------- INPUT PREPARATION ----------------
def prepare_inputs(text, max_length=128, char_max_len=20):

    # 🔥 APPLY DEFENSE BEFORE TOKENIZATION
    text = process_text(text)

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    char_ids = []
    for token in tokens:

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            char_ids.append([0] * char_max_len)
            continue

        token = token.replace("##", "")
        chars = list(token)

        ids = [char_vocab.get(c, 0) for c in chars[:char_max_len]]
        ids += [0] * (char_max_len - len(ids))
        char_ids.append(ids)

    char_ids = torch.tensor([char_ids])

    return input_ids, char_ids, attention_mask


# ---------------- EMBEDDING ----------------
def get_sentence_embedding(output, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    summed = torch.sum(output * mask, dim=1)
    count = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / count


# ---------------- 🔥 ADD THIS (IMPORTANT FOR PIPELINE) ----------------
def compute_similarity(model, text1, text2):

    model.eval()

    with torch.no_grad():

        # Process both texts
        input_ids1, char_ids1, mask1 = prepare_inputs(text1)
        input_ids2, char_ids2, mask2 = prepare_inputs(text2)

        out1 = model(input_ids1, char_ids1, mask1)
        out2 = model(input_ids2, char_ids2, mask2)

        emb1 = get_sentence_embedding(out1, mask1)
        emb2 = get_sentence_embedding(out2, mask2)

        sim = torch.nn.functional.cosine_similarity(emb1, emb2)

    return sim.item()