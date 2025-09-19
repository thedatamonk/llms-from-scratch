"""
From the scratch implementation of the original transformer model proposed 
in the paper "Attention is All You Need".

Reference: https://huggingface.co/datasets/bird-of-paradise/transformer-from-scratch-tutorial/blob/main/Transformer_Implementation_Tutorial.ipynb
"""

import math
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingWithProjection(nn.Module):
    def __init__(self, vocab_size, d_embed, d_model, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.projection = nn.Linear(self.d_embed, self.d_model)

        # What is this for?
        self.scaling = float(math.sqrt(self.d_model))

        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    # NOTE: Why the method is a staticmethod?
    # NOTE: Need to deeply understand the logic of this from the original paper and simplify this logic if possible
    @staticmethod
    def create_positional_encoding(seq_length, d_model, batch_size=1):

        # For each position in the sequence, we need to create an embedding
        # So let's create a placeholder vector of dimension (seq_length, 1) for that
        position = torch.arange(seq_length).unsqueeze(1).float()

        # Create dimension indices: [1, d_model//2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Create empty tensor: [seq_length, d_model]
        pe = torch.zeros(seq_length, d_model)
        
        # Compute sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and expand: [batch_size, seq_length, d_model]
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pe
    
    def forward(self, x):
        assert x.dtype == torch.long, f"Input tensor must have dtype torch.long, got {x.dtype}"
        batch_size, seq_length = x.size() # [batch, seq_length]

        # token embedding
        token_embedding = self.embedding(x)                                                            #[2, 16, 1024]     
        # project the scaled token embedding to the d_model space
        token_embedding =  self.projection(token_embedding) * self.scaling                             #[2, 16, 768]

        # add positional encodings to projected, 
        # scaled embeddings before applying layer norm and dropout.
        positional_encoding = self.create_positional_encoding(seq_length, self.d_model, batch_size)    #[2, 16, 768]
        
        # In addition, we apply dropout to the sums of the embeddings 
        # in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1.
        normalized_sum = self.layernorm(token_embedding + positional_encoding)
        final_output = self.dropout(normalized_sum)
        return final_output
    

class TransformerAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1, bias=True): # infer d_k, d_v, d_q from d_model
        super().__init__()
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        self.d_model = d_model
        self.num_head = num_head
        self.d_head=d_model//num_head
        self.dropout_rate = dropout

        # linear transformations
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # NOTE: Need to understand this. Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.d_head))

    def forward(self, sequence, key_value_states = None, att_mask=None):
        # NOTE: Is key_value_states the KV cache?
        """Input shape: [batch_size, seq_len, d_model=num_head * d_head]"""
        batch_size, seq_len, model_dim = sequence.size()

        # Check only critical input dimensions
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn't match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
            f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.d_model}"
        

        # NOTE: What does this note mean?
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # Linear projections and reshape for multi-head
        Q_state = self.q_proj(sequence)
        if is_cross_attention:
            kv_seq_len = key_value_states.size(1)
            K_state = self.k_proj(key_value_states)
            V_state = self.v_proj(key_value_states)
        else:
            kv_seq_len = seq_len
            K_state = self.k_proj(sequence)
            V_state = self.v_proj(sequence)

        # [batch_size, self.num_head, seq_len, self.d_head]
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head).transpose(1,2)
        
        # in cross-attention, key/value sequence length might be different from query sequence length
        # NOTE: But why?
        K_state = K_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)
        V_state = V_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)

        # Scale Q by 1/sqrt(d_k)
        Q_state = Q_state * self.scaler

        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2)) 

        # apply attention mask to attention matrix
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")
    
        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask
        

        # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim = -1)

        # apply drop out to attention score
        att_score = self.dropout(att_score)
    
        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)

        # concatinate all attention heads
        att_output = att_output.transpose(1, 2)
        att_output = att_output.contiguous().view(batch_size, seq_len, self.num_head*self.d_head)

        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), f"Final output shape {att_output.size()} incorrect"

        return att_output



class FFN(nn.Module):
    """
    Position-wise Feed-Forward Networks
    """
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.d_model=d_model
        self.d_ff= d_ff
        
        # Linear transformation y = xW+b
        self.fc1 = nn.Linear(self.d_model, self.d_ff, bias = True)
        self.fc2 = nn.Linear(self.d_ff, self.d_model, bias = True)

        # for potential speed up
        # Pre-normalize the weights (can help with training stability)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, input):
        # check input and first FF layer dimension matching
        batch_size, seq_length, d_input = input.size()
        assert self.d_model == d_input, "d_model must be the same dimension as the input"

        f1 = F.relu(self.fc1(input))
        f2 =  self.fc2(f1)

        return f2
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_head, dropout=0.1, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # attention sublayer
        self.att = TransformerAttention(
            d_model = d_model,
            num_head = num_head,
            dropout = dropout,
            bias = bias
        )

        # FFN sublayer
        self.ffn = FFN(
            d_model = d_model,
            d_ff = d_ff
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # layer-normalization layer
        self.LayerNorm_att = nn.LayerNorm(self.d_model)
        self.LayerNorm_ffn = nn.LayerNorm(self.d_model)
    
    def forward(self, embed_input, padding_mask=None):
        batch_size, seq_len, _ = embed_input.size()
        ## First sublayer: self attention 
        att_sublayer = self.att(sequence = embed_input, key_value_states = None, att_mask = padding_mask)  # [batch_size, sequence_length, d_model]

        # apply dropout before layer normalization for each sublayer
        att_sublayer = self.dropout(att_sublayer)
        # Residual layer normalization
        att_normalized = self.LayerNorm_att(embed_input + att_sublayer)           # [batch_size, sequence_length, d_model]

        ## Second sublayer: FFN
        ffn_sublayer = self.ffn(att_normalized)                                   # [batch_size, sequence_length, d_model]
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized + ffn_sublayer )       # [batch_size, sequence_length, d_model]
    

        return ffn_normalized
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_head, dropout=0.1, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # attention sublayer
        self.att = TransformerAttention(
            d_model = d_model,
            num_head = num_head,
            dropout = dropout,
            bias = bias
        )

        # FFN sublayer
        self.ffn = FFN(
            d_model = d_model,
            d_ff = d_ff
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # layer-normalization layer
        self.LayerNorm_att1 = nn.LayerNorm(self.d_model)
        self.LayerNorm_att2 = nn.LayerNorm(self.d_model)
        self.LayerNorm_ffn = nn.LayerNorm(self.d_model)

    @staticmethod
    def create_causal_mask(seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, embed_input, cross_input, padding_mask=None):
        """
        Args:
        embed_input: Decoder input sequence [batch_size, seq_len, d_model]
        cross_input: Encoder output sequence [batch_size, encoder_seq_len, d_model]
        casual_attention_mask: Causal mask for self-attention [batch_size, seq_len, seq_len]
        padding_mask: Padding mask for cross-attention [batch_size, seq_len, encoder_seq_len]
        Returns:
        Tensor: Decoded output [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = embed_input.size()

        assert embed_input.size(-1) == self.d_model, f"Input dimension {embed_input.size(-1)} doesn't match model dimension {self.d_model}"
        assert cross_input.size(-1) == self.d_model, "Encoder output dimension doesn't match model dimension"

        # Generate and expand causal mask for self-attention
        causal_mask = self.create_causal_mask(seq_len).to(embed_input.device)  # [seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]

        att_sublayer1 = self.att(sequence = embed_input, key_value_states = None, att_mask = causal_mask)  # [batch_size, num_head, sequence_length, d_model]
        
        # apply dropout before layer normalization for each sublayer
        att_sublayer1 = self.dropout(att_sublayer1)
        # Residual layer normalization
        att_normalized1 = self.LayerNorm_att1(embed_input + att_sublayer1)


        ## Second sublayer: cross attention
        # Query from the output of previous attention output, or training data
        # Key, Value from output of Encoder of the same layer
        att_sublayer2 = self.att(sequence = att_normalized1, key_value_states = cross_input, 
                                att_mask = padding_mask)  # [batch_size, sequence_length, d_model]
        

        # apply dropout before layer normalization for each sublayer
        att_sublayer2 = self.dropout(att_sublayer2)
        # Residual layer normalization
        att_normalized2 = self.LayerNorm_att2(att_normalized1 + att_sublayer2)           # [batch_size, sequence_length, d_model]


        # Third sublayer: FFN
        ffn_sublayer = self.ffn(att_normalized2)                                   # [batch_size, sequence_length, d_model]
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized2 + ffn_sublayer )       # [batch_size, sequence_length, d_model]
    

        return ffn_normalized

class TransformerEncoderDecoder(nn.Module):
    """
    Encoder-Decoder stack of the Transformer
    Sublayers:  Encoder x 6
                Decoder x 6
    Args:
            d_model: 512 model hidden dimension
            d_embed: 512 embedding dimension, same as d_model in transformer framework
            d_ff: 2048 hidden dimension of the feed forward network
            num_head: 8 Number of attention heads.
            dropout:  0.1 dropout rate 
            
            bias: Whether to include bias in linear projections.
              
    """
    def __init__(
        self, num_layer,
        d_model, d_ff,
        num_head, dropout=0.1,
        bias=True
    ):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_head
        self.dropout = dropout
        self.bias = bias
        
        # Encoder stack
        self.encoder_stack = nn.ModuleList([EncoderBlock(
                                        d_model = self.d_model, 
                                        d_ff = self.d_ff,
                                        num_head = self.num_head, 
                                        dropout = self.dropout,
                                        bias = self.bias) for _ in range(self.num_layer)])

        # Decoder stack
        self.decoder_stack = nn.ModuleList([DecoderBlock(
                                        d_model = self.d_model, 
                                        d_ff = self.d_ff,
                                        num_head = self.num_head, 
                                        dropout = self.dropout,
                                        bias = self.bias) for _ in range(self.num_layer)])

    
    def forward(self, embed_encoder_input, embed_decoder_input, padding_mask=None):
        # Process through all encoder layers first
        encoder_output = embed_encoder_input
        for encoder in self.encoder_stack:
            encoder_output = encoder(encoder_output, padding_mask)
        
        # Use final encoder output for all decoder layers
        decoder_output = embed_decoder_input
        for decoder in self.decoder_stack:
            decoder_output = decoder(decoder_output, encoder_output, padding_mask)
        
        return decoder_output

class Transformer(nn.Module):
    def __init__(
        self, 
        num_layer,
        d_model, d_embed, d_ff,
        num_head,
        src_vocab_size, 
        tgt_vocab_size,
        max_position_embeddings=512,
        dropout=0.1,
        bias=True
    ):
        super().__init__()
        
        self.tgt_vocab_size = tgt_vocab_size
        
        # Source and target embeddings
        self.src_embedding = EmbeddingWithProjection(
            vocab_size=src_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        self.tgt_embedding = EmbeddingWithProjection(
            vocab_size=tgt_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        # Encoder-Decoder stack
        self.encoder_decoder = TransformerEncoderDecoder(
            num_layer=num_layer,
            d_model=d_model,
            d_ff=d_ff,
            num_head=num_head,
            dropout=dropout,
            bias=bias
        )
        
        # Output projection and softmax
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def shift_target_right(self, tgt_tokens):
        # Shift target tokens right by padding with zeros at the beginning
        batch_size, seq_len = tgt_tokens.size()
        
        # Create start token (zeros)
        start_tokens = torch.zeros(batch_size, 1, dtype=tgt_tokens.dtype, device=tgt_tokens.device)
        
        # Concatenate start token and remove last token
        shifted_tokens = torch.cat([start_tokens, tgt_tokens[:, :-1]], dim=1)
        
        return shifted_tokens
        
    def forward(self, src_tokens, tgt_tokens, padding_mask=None):
        """
        Args:
            src_tokens: source sequence [batch_size, src_len]
            tgt_tokens: target sequence [batch_size, tgt_len]
            padding_mask: padding mask [batch_size, 1, 1, seq_len]
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size] log probabilities
        """
        # Shift target tokens right for teacher forcing
        shifted_tgt_tokens = self.shift_target_right(tgt_tokens)
        
        # Embed source and target sequences
        src_embedding = self.src_embedding(src_tokens)
        tgt_embedding = self.tgt_embedding(shifted_tgt_tokens)
        
        # Pass through encoder-decoder stack
        decoder_output = self.encoder_decoder(
            embed_encoder_input=src_embedding,
            embed_decoder_input=tgt_embedding,
            padding_mask=padding_mask
        )
        
        # Project to vocabulary size and apply log softmax
        logits = self.output_projection(decoder_output)
        log_probs = self.softmax(logits)
        
        return log_probs

def create_padding_mask(batch_size: int, sequence_length: int):
    """
    Creates a boolean padding mask.

    Args:
        batch_size (int): The number of sequences in the batch.
        sequence_length (int): The maximum length of each sequence.
        valid_lengths (list[int]): A list containing the true length of each sequence.

    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, sequence_length) with
                      `True` for valid tokens and `False` for padding tokens.
    """
    # Initialize a mask of all True values
    mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    valid_lengths = torch.randint(1, sequence_length + 1 // 2, (batch_size,))

    # Iterate through the batch and set padding tokens to False
    for i in range(batch_size):
        mask[i, :valid_lengths[i]] = True

    return mask

def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculates the total size of a PyTorch model in megabytes (MB).

    This function iterates through all the parameters and buffers of the model,
    determines the number of elements and the size of each element's data type,
    and then sums them up to get the total size in bytes. Finally, it converts
    the result to megabytes.

    Args:
        model (nn.Module): The PyTorch model to be measured.

    Returns:
        float: The total size of the model in megabytes.
    """
    total_size_bytes = 0
    # Add parameter sizes
    for param in model.parameters():
        total_size_bytes += param.numel() * param.element_size()
    
    # Add buffer sizes (e.g., BatchNorm running_mean and running_var)
    for buffer in model.buffers():
        total_size_bytes += buffer.numel() * buffer.element_size()
    
    # Convert bytes to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    return total_size_mb

if __name__ == "__main__":
    # batch_size = 2
    # sequence_length = 16
    # input_tensor = torch.randint(0, 100, (batch_size, sequence_length))  # (batch_size, seq_length)

    # print (f"Input tensor shape: {input_tensor.shape}")
    # embed_model = EmbeddingWithProjection(vocab_size=100, d_embed=1024, d_model=768)
    # embed_output = embed_model(input_tensor)
    # print(f"Embedding output shape: {embed_output.shape}")

    
    # att_model = TransformerAttention(d_model=768, num_head=4)
    # casual_att_mask = torch.triu(torch.full((sequence_length, sequence_length), float('-inf')), diagonal=1)
    # print (f"Casual attention mask shape: {casual_att_mask.shape}")
    # causal_attn_output = att_model(sequence=embed_output, key_value_states=None, att_mask=casual_att_mask)
    # print (f"Causal attention output shape: {causal_attn_output.shape}")

    # Simulate a batch with different sequence lengths
    # full_padding_mask = create_padding_mask(sequence_length)

    # attention_mask = torch.where(full_padding_mask, 0.0, float('-inf'))
    # print (f"Casual attention mask shape: {casual_att_mask.shape}")
    # full_attn_output = att_model(sequence=embed_output, key_value_states=None, att_mask=attention_mask)
    # print (f"Full attention output shape: {full_attn_output.shape}")

    # Encoder block
    # encoder_block = EncoderBlock(d_model=768, d_ff=2048, num_head=4, dropout=0.2, bias=True)
    # print (encoder_block)


    # decoder_block = DecoderBlock(d_model=768, d_ff=2048, num_head=4, dropout=0.2, bias=True)
    # print (decoder_block)

    transformer_model = Transformer(
        num_layer=8,
        d_model=512, d_embed=4096, d_ff=2048, num_head=8,
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        max_position_embeddings=512,
        dropout=0.1,
        bias=True
    )
    print (transformer_model)

    total_params = sum(p.numel() for p in transformer_model.parameters())

    print(f"Total number of parameters: {total_params}")

    print (f"Total model size (MB): {get_model_size_mb(transformer_model):.2f} MB")
