# imports

import re
import string
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# Initialize tokenizer and prepare stopwords
tokenizer = get_tokenizer("basic_english")
stop_words = stopwords.words("english")

# Functions
def remove_punctuations(text: str) -> str:
    """Remove punctuations from a text.

    Args:
        text (str): Text.
    Returns:
        str: Text with removed punctuations.
    """

    pattern = f"[{re.escape(string.punctuation)}]"
    return re.sub(pattern, " ", text)


def remove_numbers(text: str) -> str:
    """Remove numbers from a text.

    Args:
        text (str): Text.
    Returns:
        str: Text with numbers punctuations.
    """

    pattern = r"[0-9]"
    return re.sub(pattern, " ", text)


def remove_confidential_information(text: str) -> str:
    """Remove confidential information from a text.

    Args:
        text (str): Text.
    Returns:
        str: Text with removed confidential information.
    """

    pattern = r"\b[Xx]{1,}\b"
    return re.sub(pattern, " ", text)


def remove_extra_spaces(text: str) -> str:
    """Remove extra spaces or new lines from a text.

    Args:
        text (str): Text.
    Returns:
        str: Text with removed extra spaces or new lines.
    """
    
    pattern = r"\s+"
    return re.sub(pattern, " ", text)


def remove_stopwords(text: str) -> str:
    """Remove stop words from text

    Args:
        text (str): Text.
    Returns:
        str: Text with stop words removed.
    """

    tokens = tokenizer(text)
    return " ".join([token for token in tokens if token not in stop_words])


# source: https://www.ibm.com/topics/stemming-lemmatization#:~:text=The%20practical%20distinction%20between%20stemming,be%20found%20in%20the%20dictionary.
def get_wordnet_pos(tag: str) -> str:
    """Return wordnet constant value to do lemmatization based on their input word tag

    Args:
        tag (str): Tag name.
    Returns:
        str: Constant value for wordnet lemmatization.
    """

    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(text: str) -> str:
    """Perform lemmatization using WordNetLemmatizer

    Args:
        tokens (str): Text.
    Returns:
        str: Lemmatized text.
    """
    
    tokens = tokenizer(text)
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags])


def pad_sequence(tokens: list, max_length: int, post: bool = True) -> np.array:
    """Perform zero padding before or after the sequence.

    Args:
        tokens (str): Text.
    Returns:
        str: Padded sequences.
    """
    
    padded_tokens = None
    if len(tokens) < max_length:
        zeros = list(np.zeros(max_length - len(tokens)))
        if post:
            padded_tokens = tokens + zeros  # Add zeros after the seqeuence
        else:
            padded_tokens = zeros + tokens  # Add zeros before the seqeuence
    else:
        padded_tokens = tokens[:max_length]
    return padded_tokens


def generate_vocabulary(texts: list) -> Vocab:
    """Generates a vocabulary from a list of texts.

    Args:
        texts (list): A list of text strings to build the vocabulary from.
    Returns:
        Vocab: A vocabulary object containing the tokens and their corresponding indices.
    """

    def yield_tokens(texts: list):
        for text in texts:
            yield tokenizer(text.strip())

    # Generate vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(texts), min_freq=2, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def calculate_max_length_sequence(texts: list) -> int:
    """Calculates the maximum length of token sequences from a list of texts.

    Args:
        texts (list): Texts.
    Returns:
        int: The maximum length of the tokenized sequences.
    """
    max_length = 0
    for text in texts:
        max_length = max(max_length, len(tokenizer(text)))
    return max_length
