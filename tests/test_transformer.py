# We will use Pytest to test the Embedding class and utilities
import unittest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from archs.transformer import EmbeddingWithProjection, EncoderBlock, DecoderBlock, TransformerEncoderDecoder, Transformer

torch.manual_seed(42)

class TestEmbeddingWithProjection(unittest.TestCase):

    # This method runs before each test method
    def setUp(self):
        self.seq_length = 16
        self.batch_size = 2
        self.embed_dim = 768
        self.proj_dim = 512
        self.vocab_size = 500
        self.input_tensor = torch.randint(0, 100, (self.batch_size, self.seq_length))  # (batch_size, seq_length)

        self.model = EmbeddingWithProjection(vocab_size=self.vocab_size, d_embed=self.embed_dim, d_model=self.proj_dim)


    # Test case 1: Check the output shape
    def test_output_shape(self):
        output = self.model(self.input_tensor)
        # We use a built-in unittest assertion method
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.proj_dim))

    # Test case 2: Check if the forward pass runs without error
    def test_forward_pass_no_error(self):
        try:
            self.model(self.input_tensor)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

    # Test case 3: Check if the model is trainable (requires grad)
    def test_model_is_trainable(self):
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

    # This method runs after each test method
    def tearDown(self):
        del self.model
        del self.input_tensor
        torch.cuda.empty_cache() # Useful if you are testing on a GPU

class TestEncoderBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 20
        self.d_model = 512
        self.d_ff = 2048
        self.num_head = 8
    
        # Initialise the transformer encoder
        self.encoder = EncoderBlock(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_head=self.num_head,
            dropout=0.1,
            bias=True
        )

        # Set to evaluation model to disable dropout
        self.encoder.eval()

        # create input sequence
        self.input_sequence = torch.ones(self.batch_size, self.seq_length, self.d_model)
        self.cross_sequence = torch.ones(self.batch_size, self.seq_length, self.d_model) * 0.5 # 0.5 is just for ensuring that input and cross sequences are different

        # create attention mask
        self.attention_mask = torch.ones(self.batch_size, self.seq_length)
        self.attention_mask[:, 15:] = 0 # mask last 5 positions
        self.attention_mask = self.attention_mask.unsqueeze(1).unsqueeze(3)

        # NOTE: What is this for ? store attention patterns
        self.attention_patterns = []

        # Define hook to capture attention scores
        def attention_hook(module, input, output):
            # We want to capture the attention scores before they're processed further
            # This assumes your attention module returns the attention scores
            self.attention_patterns.append(output)

        
        # add the hook to the attention sublayer of the encoder block
        self.encoder.att.register_forward_hook(attention_hook)

    def test_forward_pass(self):
        try:

            # First forward pass with mask
            with torch.no_grad():
                output_masked = self.encoder(self.input_sequence, self.attention_mask)
                # masked_attention = self.attention_patterns[0]

            # self.attention_patterns.clear()  # Clear previous patterns
        
            # Basic shape tests
            expected_shape = (self.batch_size, self.seq_length, self.d_model)
            self.assertEqual(output_masked.shape, expected_shape, msg=f"Expected output shape {expected_shape}, got {output_masked.shape}")

            
            # Second forward pass without mask
            with torch.no_grad():
                output_unmasked = self.encoder(self.input_sequence)  # No mask
                # unmasked_attention = self.attention_patterns[0]
            
            # self.attention_patterns.clear()  # Clear previous patterns

            # Check if masking worked. We expect the masked positions to have different values.
            # Check specific positions where the mask was applied (e.g., last 5 positions).
            # We need to reshape the mask to match the attention tensor.
            # mask_reshaped = self.attention_mask.squeeze(1).squeeze(2).bool()

            # Check if the values at masked positions are different
            # A simple way to check is to find a position that was masked and ensure the values differ.
            # This is a good sanity check.
            # diff = torch.abs(masked_attention - unmasked_attention)

            # Find positions where the mask was applied (mask value is 0)
            # We expect a non-zero difference at these positions
            # masked_positions_exist = torch.any(diff[~mask_reshaped] > 1e-6)

            # self.assertTrue(masked_positions_exist, msg="Masking does not seem to be working. The attention scores at masked positions are identical.")
            self.assertTrue(torch.isfinite(output_masked).all(), msg="Output contains non-finite values (NaN or Inf).")
            self.assertTrue(torch.isfinite(output_unmasked).all(), msg="Output contains non-finite values (NaN or Inf).")


        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

        finally:
            self.encoder.att._forward_hooks.clear()

class TestDecoderBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.encoder_seq_length = 22 # NOTE: Still need to understand what is encoder_seq_length and why is it different from seq_length
        self.seq_length = 20
        self.d_model = 512
        self.d_ff = 2048
        self.num_head = 8

        self.decoder = DecoderBlock(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_head=self.num_head,
            dropout=0.1,
            bias=True
        )

        self.decoder.eval()

        # Create input sequences
        self.decoder_input = torch.randn(self.batch_size, self.seq_length, self.d_model)
        self.encoder_output = torch.randn(self.batch_size, self.encoder_seq_length, self.d_model)

        # Create padding mask for encoder outputs
        self.padding_mask = torch.ones(self.batch_size, self.seq_length, self.encoder_seq_length)
        self.padding_mask[:, :, 18:] = 0  # Mask last 4 positions of encoder output
        self.padding_mask = self.padding_mask.unsqueeze(1)  # Add head dimension

        # Store attention scores
        attention_scores = []
        
        # Define hook to capture attention scores before softmax
        def attention_hook(module, input, output):
            if not attention_scores:  # Only store first layer's patterns
                # Assuming attention scores are computed before this hook
                attention_scores.append(module.att_matrix.detach())  # You might need to modify this based on your attention implementation
        
        # Register hook on the attention layer
        self.decoder.att.register_forward_hook(attention_hook)

    def test_forward_pass(self):

        try:

            # Perform forward pass
            with torch.no_grad():
                output = self.decoder(self.decoder_input, self.encoder_output, self.padding_mask)

            # Basic shape tests
            expected_shape = (self.batch_size, self.seq_length, self.d_model)
            self.assertEqual(output.shape, expected_shape, msg=f"Expected output shape {expected_shape}, got {output.shape}")

            # Print output statistics
            print("\nOutput Statistics:")
            print(f"Mean: {output.mean():.4f}")
            print(f"Std: {output.std():.4f}")
            print(f"Min: {output.min():.4f}")
            print(f"Max: {output.max():.4f}")

            # Test shape preservation
            print("\nShape Analysis:")
            print(f"Input shape: {self.decoder_input.shape}")
            print(f"Output shape: {output.shape}")
            self.assertEqual(self.decoder_input.shape, output.shape, msg="Input and output shapes do not match.")
            
            # Check for any NaN or infinite values
            self.assertTrue(torch.isfinite(output).all(), msg="Output contains NaN or infinite values")
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

        finally:
            self.decoder.att._forward_hooks.clear()

class TestEncoderDecoderStack(unittest.TestCase):
    def setUp(self):
        # Test parameters
        self.batch_size = 8
        self.seq_length = 10
        self.d_model = 512
        self.d_ff = 2048
        self.num_heads = 8
        self.num_layers = 6


        # initialise encoder and decoder stacks
        self.transformer = TransformerEncoderDecoder(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_head=self.num_heads,
            num_layer=self.num_layers,
            dropout=0.1,
            bias=True,
        )

        self.transformer.eval()

        # create input sequences
        self.encoder_input = torch.randn(self.batch_size, self.seq_length, self.d_model)
        self.decoder_input = torch.randn(self.batch_size, self.seq_length, self.d_model)

        # create padding mask
        self.padding_mask = torch.ones(self.batch_size, self.seq_length)
        self.padding_mask[:, -2:] = 0  # Mask last 2 positions
        self.padding_mask = self.padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

        # store intermediate outputs
        intermediate_outputs = []

        def hook_fn(module, input, output):
            intermediate_outputs.append(output.detach())
        
        # Register hooks to capture outputs from each encoder and decoder layer
        self.encoder_handles = []
        self.decoder_handles = []

        for i, (encoder, decoder) in enumerate(zip(self.transformer.encoder_stack, self.transformer.decoder_stack)):
            encoder_handle = encoder.register_forward_hook(lambda m, i, o, layer=i: print(f"\nEncoder Layer {layer} shape:", o.shape))
            self.encoder_handles.append(encoder_handle)
            decoder_handle = decoder.register_forward_hook(lambda m, i, o, layer=i: print(f"Decoder Layer {layer} shape:", o.shape))
            self.decoder_handles.append(decoder_handle)

    def test_forward_pass(self):
        try:
            with torch.no_grad():
                output = self.transformer(self.encoder_input, self.decoder_input, self.padding_mask)
            
            expected_shape = (self.batch_size, self.seq_length, self.d_model)
            self.assertEqual(output.shape, expected_shape, msg=f"Expected output shape {expected_shape}, got {output.shape}")

            # Print output statistics
            print("\nFinal Output Statistics:")
            print(f"Mean: {output.mean():.4f}")
            print(f"Std: {output.std():.4f}")
            print(f"Min: {output.min():.4f}")
            print(f"Max: {output.max():.4f}")
            
            # Verify shape preservation through layers
            print("\nShape Preservation Check:")
            print(f"Input shapes - Encoder: {self.encoder_input.shape}, Decoder: {self.decoder_input.shape}")
            print(f"Output shape: {output.shape}")
            
            # Check for any NaN or infinite values
            self.assertTrue(torch.isfinite(output).all(), msg="Output contains NaN or infinite values")
            
            # Verify that output is different from input (transformation happened)
            input_output_diff = (output - self.decoder_input).abs().mean()
            print(f"\nMean absolute difference between input and output: {input_output_diff:.4f}")
            self.assertTrue(input_output_diff > 1e-3, msg="Output is too similar to input, transformation might not have occurred.")
            
            # Check if model parameters were used
            total_params = sum(p.numel() for p in self.transformer.parameters())
            print(f"\nTotal number of parameters: {total_params:,}")
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")
        finally:
            # remove all the hooks
            for handle in self.encoder_handles + self.decoder_handles:
                handle.remove()
            # The lists containing the handles can also be cleared if needed
            self.encoder_handles.clear()
            self.decoder_handles.clear()

class TestCompleteTransformer(unittest.TestCase):
    def setUp(self):
        # Configuration
        self.d_model = 768
        self.d_embed = 1024
        self.d_ff = 2048
        self.num_heads = 8
        self.num_layers = 6
        self.max_position_embeddings = 512

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", 
                                            use_fast=True, 
                                            use_multiprocessing=False)
        self.vocab_size = self.tokenizer.vocab_size


        # Create sample source and target sequences
        self.src_sequences = [
            "I've been waiting for a HuggingFace course my whole life.",
            "So have I!"
        ]
        # Pretend these are translations
        self.tgt_sequences = [
            "J'ai attendu un cours HuggingFace toute ma vie.",
            "Moi aussi!"
        ]

        # Tokenize source and target sequences
        self.src_inputs = self.tokenizer(self.src_sequences, truncation=True, padding="longest", return_tensors="pt")
        self.tgt_inputs = self.tokenizer(self.tgt_sequences, truncation=True, padding="longest", return_tensors="pt")

        # Create transformer model
        self.transformer = Transformer(
            num_layer=self.num_layers,
            d_model=self.d_model,
            d_embed=self.d_embed,
            d_ff=self.d_ff,
            num_head=self.num_heads,
            src_vocab_size=self.vocab_size,
            tgt_vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings
        )
        
        # Set to eval mode
        self.transformer.eval()

        # Create padding mask from attention mask
        self.padding_mask = self.src_inputs['attention_mask'].unsqueeze(1).unsqueeze(2)
    
        print("\nInput Shapes:")
        print(f"Source tokens: {self.src_inputs['input_ids'].shape}")
        print(f"Target tokens: {self.tgt_inputs['input_ids'].shape}")

    def test_forward_pass(self):
        try:
            with torch.no_grad():
                output = self.transformer(
                            src_tokens=self.src_inputs['input_ids'],
                            tgt_tokens=self.tgt_inputs['input_ids'],
                            padding_mask=self.padding_mask
                        )
                
            print("\nOutput Analysis:")
            print(f"Output shape: {output.shape}")  # Should be [batch_size, tgt_len, vocab_size]

            # Verify output is proper probability distribution
            print("\nProbability Distribution Check:")
            self.assertTrue(torch.allclose(output.exp().sum(dim=-1), torch.ones_like(output.exp().sum(dim=-1))), msg="Output probabilities do not sum to 1.")
            print(f"Max probability: {output.exp().max().item():.4f}")
            print(f"Min probability: {output.exp().min().item():.4f}")
            
            # Check if we can get predictions
            predictions = output.argmax(dim=-1)
            print("\nSample Predictions:")
            print("Original target:")
            print(self.tgt_sequences[0])
            print("\nModel output (decoded):")
            print(self.tokenizer.decode(predictions[0]))

            # Test backward pass
            self.transformer.train()
            output = self.transformer(
                src_tokens=self.src_inputs['input_ids'],
                tgt_tokens=self.tgt_inputs['input_ids'],
                padding_mask=self.padding_mask
            )

            # calculate cross entropy loss
            loss = F.nll_loss(
                output.view(-1, self.vocab_size),
                self.tgt_inputs['input_ids'].view(-1),
            )

            loss.backward()

            # Check if loss is a finite number
            self.assertTrue(torch.isfinite(loss).all(), msg="Loss is not a finite number")
            print(f"\nLoss: {loss.item():.4f}")

            # Check gradients
            has_gradients = all(p.grad is not None for p in self.transformer.parameters())
            self.assertTrue(has_gradients, msg="Not all model parameters have gradients after backward pass")
            print("All model parameters have gradients after backward pass.")
        except Exception as e:
            self.fail(f"Forward or backward pass raised an exception: {e}")
