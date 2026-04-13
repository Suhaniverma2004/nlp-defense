import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from src.char_vocab import build_char_vocab

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
char_vocab = build_char_vocab()


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


def prepare_inputs(text, max_length=128, char_max_len=20):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"]
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
    return input_ids, char_ids


def get_sentence_embedding(output, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    summed = torch.sum(output * mask, dim=1)
    count = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / count