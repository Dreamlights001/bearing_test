import torch
import torch.nn as nn
import math
from data_loader import NUM_CLASSES, SEQUENCE_LENGTH

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=SEQUENCE_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [seq_len, batch_size, d_model]"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BearingTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=3, 
                 dim_feedforward=256, dropout=0.1, num_classes=NUM_CLASSES):
        super(BearingTransformer, self).__init__()
        self.d_model = d_model
        
        # Input embedding: from 1 feature to d_model
        # The input data is [batch_size, seq_len, input_dim]
        # We want it to be [seq_len, batch_size, d_model] for the transformer
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=SEQUENCE_LENGTH)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=False # Expects (seq_len, batch, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Classifier head
        # The output of the transformer encoder is [seq_len, batch_size, d_model]
        # We can take the mean over the sequence length or the output of the [CLS] token if we had one.
        # Here, let's take the mean.
        self.fc_out = nn.Linear(d_model, num_classes)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """ 
        src: [batch_size, seq_len, input_dim] (e.g., [32, 1024, 1])
        """
        # Embedding and positional encoding
        # src shape: [batch_size, seq_len, input_dim]
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)
        # embedded_src shape: [batch_size, seq_len, d_model]
        
        # Transformer expects [seq_len, batch_size, d_model]
        embedded_src = embedded_src.permute(1, 0, 2) 
        # embedded_src shape: [seq_len, batch_size, d_model]
        
        pos_encoded_src = self.pos_encoder(embedded_src)
        # pos_encoded_src shape: [seq_len, batch_size, d_model]
        
        # Transformer encoder
        transformer_output = self.transformer_encoder(pos_encoded_src)
        # transformer_output shape: [seq_len, batch_size, d_model]
        
        # Classification
        # We can average the output features over the sequence length
        output_mean = transformer_output.mean(dim=0) # [batch_size, d_model]
        # Or take the output of the first token (if it's like a CLS token)
        # output_cls = transformer_output[0, :, :] # [batch_size, d_model]
        
        logits = self.fc_out(output_mean)
        # logits shape: [batch_size, num_classes]
        return logits

if __name__ == '__main__':
    # Example Usage
    batch_size = 32
    seq_len = SEQUENCE_LENGTH # from data_loader
    input_dim_feat = 1 # 1 feature (amplitude)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, input_dim_feat)
    print(f"Input shape: {dummy_input.shape}")

    # Initialize the model
    model = BearingTransformer(input_dim=input_dim_feat, 
                               d_model=64, 
                               nhead=4, 
                               num_encoder_layers=2, 
                               dim_feedforward=128, 
                               dropout=0.1, 
                               num_classes=NUM_CLASSES)
    
    # Get model output
    output_logits = model(dummy_input)
    print(f"Output logits shape: {output_logits.shape}") # Expected: [batch_size, num_classes]

    # Check number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Test with different parameters
    model_large = BearingTransformer(input_dim=input_dim_feat, 
                                   d_model=128, 
                                   nhead=8, 
                                   num_encoder_layers=4, 
                                   dim_feedforward=512, 
                                   dropout=0.2, 
                                   num_classes=NUM_CLASSES)
    output_logits_large = model_large(dummy_input)
    print(f"Output logits shape (large model): {output_logits_large.shape}")
    num_params_large = sum(p.numel() for p in model_large.parameters() if p.requires_grad)
    print(f"Number of trainable parameters (large model): {num_params_large:,}")