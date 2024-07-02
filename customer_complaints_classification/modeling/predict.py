import time

def predict(text: str, vocab: Vocab, model: nn.Module, max_length: int) -> int:
    """
    Predicts the class label for a given text.

    Args:
        text (str): Input text to be classified.
        vocab (Vocab): Vocabulary object used to encode the text.
        model (nn.Module): Trained neural network model for prediction.
        max_length (int): Maximum length of the input text sequences.

    Returns:
        int: Predicted class label.
    """
    # Preprocess the input text:
    text = preprocess_text(text)
    
    # Encode the text using the vocabulary:
    encoded_text = vocab(tokenizer(text))
    
    # Pad the encoded text to the maximum length:
    encoded_text = pad_sequence(encoded_text, max_length)
    
    # Convert the encoded text to a tensor and add a batch dimension:
    encoded_text = torch.LongTensor(encoded_text).unsqueeze(0).to(device)
    
    # Predict the class label using the model:
    output = model(encoded_text).argmax(1)
    
    return output.item()
