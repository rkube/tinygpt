

# super simple bigram model

import torch
import torch.nn as nn

torch.manual_seed(1337)

vocab_size = 65
n_embed = 32

batch_size = 4
block_size = 8
channels = 2

x = torch.randint(vocab_size, (batch_size, block_size, channels))

token_embedding_table = nn.Embedding(vocab_size, n_embed)
position_embedding_table = nn.Embedding(block_size, n_embed)
lm_head = nn.Linear(n_embed, vocab_size)




class BigramLanguageModel(nn.Module):

    def __init__(self):
        super()._init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)

        x = tok_emb + pos_emb