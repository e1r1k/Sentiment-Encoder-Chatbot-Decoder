import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(embed_size)

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size, embed_size)
        )
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, attn_mask, padding_mask):

        n1 = self.norm1(x)
        attn_outputs, _ = self.mha(n1, n1, n1, 
                                   padding_mask, 
                                   need_weights=False, 
                                   attn_mask=attn_mask, 
                                   is_causal=True)
        x = x + self.dropout1(attn_outputs)

        n2 = self.norm2(x)
        return n2  + self.dropout2(self.mlp(n2))
        




class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        P  = torch.zeros(max_len, embed_size)

        k = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, embed_size, 2)

        denom = torch.pow(10000, i / embed_size)

        P[:, 0::2] = torch.sin(k / denom)
        P[:, 1::2] = torch.cos(k / denom)

        self.register_buffer('positional_encoding', P)

    def forward(self, x):
        positional_encoding = (self.positional_encoding[:x.size(1), :].unsqueeze(0)).to(x.device)
        return x + positional_encoding

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        matrix = torch.ones(seq_len, seq_len)
        return (torch.triu(matrix, diagonal=1)).bool()


if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

