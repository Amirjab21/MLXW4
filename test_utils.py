import torch
from dataset import __getitem__
from Transformer import generate_padding_mask

# Global variables that need to be defined
def initialize_test_environment(transformer_model, images, vocab, rev_vocab):
    """Initialize global variables needed for testing."""
    global transformer, transformed_images, vocabulary, reverse_vocab, vocab_size_final
    
    transformer = transformer_model
    transformed_images = images
    vocabulary = vocab
    reverse_vocab = rev_vocab
    vocab_size_final = len(vocabulary)

def compare_logits(image_id=None):
    """Compare logits from transformer.forward() and generate_caption's next_token_logits."""
    if image_id is None:
        image_id = torch.randint(0, len(transformed_images), (1,)).item()
    
    print(f"\nComparing logits for image {image_id}")
    print("=" * 80)
    
    # Get image data
    image_data = __getitem__(image_id, transformed_images, vocab_size_final)
    
    # Method 1: evaluate_topk approach
    print("\nMethod 1 (evaluate_topk) intermediate values:")
    output1 = transformer.forward(image_data['image'].unsqueeze(0), torch.tensor([[49406]]))  # Start token
    first_token_logits1 = output1[0, 0]  # Shape should be [vocab_size]
    
    # Method 2: generate_caption approach with detailed debugging
    print("\nMethod 2 (generate_caption) intermediate values:")
    with torch.no_grad():
        # Step 1: Image features
        image_features1 = transformer.clip.get_image_features(image_data['image'].unsqueeze(0))
        print("\nStep 1 - Image features:")
        print(f"Shape: {image_features1.shape}")
        print(f"Mean: {image_features1.mean().item():.4f}")
        print(f"Std: {image_features1.std().item():.4f}")
        
        # Step 2: Text embeddings
        caption = torch.tensor([[transformer.tokenizer.bos_token_id]])
        text_embeddings = transformer.text_embedding(caption)
        print("\nStep 2 - Text embeddings:")
        print(f"Shape: {text_embeddings.shape}")
        print(f"Mean: {text_embeddings.mean().item():.4f}")
        print(f"Std: {text_embeddings.std().item():.4f}")
        
        # Step 3: Combined sequence
        sequence = torch.cat([
            image_features1.unsqueeze(1),
            text_embeddings
        ], dim=1)
        print("\nStep 3 - Combined sequence:")
        print(f"Shape: {sequence.shape}")
        print(f"Mean: {sequence.mean().item():.4f}")
        print(f"Std: {sequence.std().item():.4f}")
        
        # Step 4: Positional encoding
        sequence = transformer.positional_encoding(sequence)
        print("\nStep 4 - After positional encoding:")
        print(f"Shape: {sequence.shape}")
        print(f"Mean: {sequence.mean().item():.4f}")
        print(f"Std: {sequence.std().item():.4f}")
        
        # Step 5: Generate mask
        mask = generate_padding_mask(caption, 49408)
        print("\nStep 5 - Attention mask:")
        print(f"Shape: {mask.shape}")
        print(f"True values: {mask.sum().item()}")
        
        # Step 6: Decoder output
        decoder_output = transformer.decoder(sequence, mask=mask)
        print("\nStep 6 - Decoder output:")
        print(f"Shape: {decoder_output.shape}")
        print(f"Mean: {decoder_output.mean().item():.4f}")
        print(f"Std: {decoder_output.std().item():.4f}")
        
        # Step 7: Get logits
        all_logits = transformer.fc(decoder_output)
        print("\nStep 7 - All logits:")
        print(f"Shape: {all_logits.shape}")
        print(f"Mean: {all_logits.mean().item():.4f}")
        print(f"Std: {all_logits.std().item():.4f}")
        
        # Extract final token logits
        first_token_logits2 = all_logits[0, -1, :]
        
        # Compare sequence of operations with transformer.forward
        print("\nComparing with transformer.forward internals:")
        # Get the same values from transformer.forward for comparison
        with torch.no_grad():
            forward_image = image_data['image'].unsqueeze(0)
            forward_token = torch.tensor([[49406]])
            
            # Get internal values from transformer.forward
            image_features2 = transformer.clip.get_image_features(forward_image)
            text_emb2 = transformer.text_embedding(forward_token)
            
            print("\nImage features comparison:")
            print(f"Max difference: {(image_features1 - image_features2).abs().max().item():.4e}")
            
            print("\nText embedding comparison:")
            print(f"Max difference: {(text_embeddings - text_emb2).abs().max().item():.4e}")
    
    # Compare final logits
    print("\nFinal logits comparison:")
    print(f"Shape of evaluate_topk logits: {first_token_logits1.shape}")
    print(f"Shape of generate_caption logits: {first_token_logits2.shape}")
    
    # Statistical comparison of final logits
    diff = first_token_logits1 - first_token_logits2
    print(f"\nLogits statistics:")
    print(f"Mean absolute difference: {torch.abs(diff).mean().item():.4f}")
    print(f"Max absolute difference: {torch.abs(diff).max().item():.4f}")
    print(f"Are logits exactly equal? {torch.allclose(first_token_logits1, first_token_logits2)}")
    print(f"Correlation coefficient: {torch.corrcoef(torch.stack([first_token_logits1, first_token_logits2]))[0,1].item():.4f}")
    
    # Show where the biggest differences are
    top_diffs_values, top_diffs_indices = torch.topk(torch.abs(diff), k=5)
    print("\nBiggest differences in final logits:")
    for idx, (index, diff_val) in enumerate(zip(top_diffs_indices, top_diffs_values)):
        token = reverse_vocab[index.item()]
        val1 = first_token_logits1[index].item()
        val2 = first_token_logits2[index].item()
        print(f"{idx+1}. Token: {token}")
        print(f"   evaluate_topk logit: {val1:.4f}")
        print(f"   generate_caption logit: {val2:.4f}")
        print(f"   absolute difference: {diff_val.item():.4f}")
        
    return {
        'image_features1': image_features1,
        'image_features2': image_features2,
        'text_embeddings1': text_embeddings,
        'text_embeddings2': text_emb2,
        'decoder_output': decoder_output,
        'all_logits': all_logits,
        'first_token_logits1': first_token_logits1,
        'first_token_logits2': first_token_logits2
    }

def test_sequence_vs_latest():
    """Compare generate_caption using whole sequence vs just latest token."""
    # Get a random image
    image_id = torch.randint(0, len(transformed_images), (1,)).item()
    image_data = __getitem__(image_id, transformed_images, vocab_size_final)
    
    print(f"\nTesting with image {image_id}")
    print("=" * 80)
    
    with torch.no_grad():
        # Get image features
        image_features = transformer.clip.get_image_features(image_data['image'].unsqueeze(0))
        
        # Initialize caption with start token
        caption = torch.tensor([[transformer.tokenizer.bos_token_id]])
        
        # Get text embeddings
        text_embeddings = transformer.text_embedding(caption)
        
        # Combine with image features
        sequence = torch.cat([
            image_features.unsqueeze(1),
            text_embeddings
        ], dim=1)
        
        # Add positional encoding
        sequence = transformer.positional_encoding(sequence)
        
        # Create mask
        mask = generate_padding_mask(caption, 49408)
        
        # Get decoder output
        decoder_output = transformer.decoder(sequence, mask=mask)
        
        # Method 1: Use whole sequence
        logits_whole = transformer.fc(decoder_output)
        first_token_logits1 = logits_whole[0, -1, :]
        
        # Method 2: Use just latest token
        logits_latest = transformer.fc(decoder_output[:, -1:, :])
        first_token_logits2 = logits_latest[0, 0, :]
        
        # Compare the results
        print("\nComparing logits:")
        print(f"Shape using whole sequence: {logits_whole.shape}")
        print(f"Shape using latest token: {logits_latest.shape}")
        
        diff = first_token_logits1 - first_token_logits2
        print(f"\nDifference statistics:")
        print(f"Mean absolute difference: {torch.abs(diff).mean().item():.4e}")
        print(f"Max absolute difference: {torch.abs(diff).max().item():.4e}")
        print(f"Are exactly equal? {torch.allclose(first_token_logits1, first_token_logits2)}")
        
        # Show top predictions from both methods
        top_k = 5
        print("\nTop predictions using whole sequence:")
        probs1 = torch.softmax(first_token_logits1, dim=-1)
        top_probs1, top_indices1 = torch.topk(probs1, k=top_k)
        for idx, (index, prob) in enumerate(zip(top_indices1, top_probs1)):
            token = reverse_vocab[index.item()]
            print(f"{idx+1}. {token}: {prob.item():.4f}")
        
        print("\nTop predictions using latest token:")
        probs2 = torch.softmax(first_token_logits2, dim=-1)
        top_probs2, top_indices2 = torch.topk(probs2, k=top_k)
        for idx, (index, prob) in enumerate(zip(top_indices2, top_probs2)):
            token = reverse_vocab[index.item()]
            print(f"{idx+1}. {token}: {prob.item():.4f}")

def test_forward_vs_separate():
    """Compare transformer.forward() vs doing steps separately."""
    # Get a random image
    image_id = torch.randint(0, len(transformed_images), (1,)).item()
    image_data = __getitem__(image_id, transformed_images, vocab_size_final)
    device = next(transformer.parameters()).device
    
    print(f"\nTesting with image {image_id}")
    print("=" * 80)
    
    with torch.no_grad():
        # Method 1: Using transformer.forward()
        input_image = image_data['image'].unsqueeze(0).to(device)
        input_text = torch.tensor([[49406]], device=device)  # Start token
        output1 = transformer.forward(input_image, input_text)
        first_token_logits1 = output1[0, 0]
        
        # Method 2: Doing steps separately
        # Get image features
        image_features = transformer.clip.get_image_features(input_image)
        
        # Get text embeddings
        text_embeddings = transformer.text_embedding(input_text)
        
        # Combine with image features
        sequence = torch.cat([
            image_features.unsqueeze(1),
            text_embeddings
        ], dim=1)
        
        # Add positional encoding
        sequence = transformer.positional_encoding(sequence)
        
        # Create mask
        mask = generate_padding_mask(input_text, 49408)
        
        # Get decoder output
        decoder_output = transformer.decoder(sequence, mask=mask)
        
        # Remove CLS token (image features) before applying fc layer
        decoder_output = decoder_output[:, 1:]
        
        # Get logits
        logits = transformer.fc(decoder_output)
        first_token_logits2 = logits[0, 0]
        
        # Compare the results
        print("\nComparing outputs:")
        print(f"Shape from forward(): {output1.shape}")
        print(f"Shape from separate steps: {logits.shape}")
        
        diff = first_token_logits1 - first_token_logits2
        print(f"\nDifference statistics:")
        print(f"Mean absolute difference: {torch.abs(diff).mean().item():.4e}")
        print(f"Max absolute difference: {torch.abs(diff).max().item():.4e}")
        print(f"Are exactly equal? {torch.allclose(first_token_logits1, first_token_logits2)}")
        
        if not torch.allclose(first_token_logits1, first_token_logits2):
            # Show top predictions from both methods
            top_k = 5
            print("\nTop predictions using forward():")
            probs1 = torch.softmax(first_token_logits1, dim=-1)
            top_probs1, top_indices1 = torch.topk(probs1, k=top_k)
            for idx, (index, prob) in enumerate(zip(top_indices1, top_probs1)):
                token = reverse_vocab[index.item()]
                print(f"{idx+1}. {token}: {prob.item():.4f}")
            
            print("\nTop predictions using separate steps:")
            probs2 = torch.softmax(first_token_logits2, dim=-1)
            top_probs2, top_indices2 = torch.topk(probs2, k=top_k)
            for idx, (index, prob) in enumerate(zip(top_indices2, top_probs2)):
                token = reverse_vocab[index.item()]
                print(f"{idx+1}. {token}: {prob.item():.4f}")
            
            # Show biggest differences
            top_diffs_values, top_diffs_indices = torch.topk(torch.abs(diff), k=5)
            print("\nBiggest differences:")
            for idx, (index, diff_val) in enumerate(zip(top_diffs_indices, top_diffs_values)):
                token = reverse_vocab[index.item()]
                val1 = first_token_logits1[index].item()
                val2 = first_token_logits2[index].item()
                print(f"{idx+1}. Token: {token}")
                print(f"   forward() logit: {val1:.4f}")
                print(f"   separate steps logit: {val2:.4f}")
                print(f"   absolute difference: {diff_val.item():.4f}")

# Initialize the test environment
initialize_test_environment(
    transformer_model=your_transformer,
    images=your_image_dataset,
    vocab=your_vocabulary,
    rev_vocab=your_reverse_vocabulary
)
