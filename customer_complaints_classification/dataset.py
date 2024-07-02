from typing import Tuple
from torch.utils.data import Dataset

class CustomerComplaintsDataset(Dataset):
    """
    A custom dataset for handling encoded customer complaints and their labels.

    Args:
        encoded_texts (list): List of encoded text sequences.
        encoded_labels (list): List of encoded labels corresponding to the text sequences.
        max_length (int): Maximum length of the text sequences.

    Attributes:
        encoded_texts (list): Stored list of encoded text sequences.
        encoded_labels (list): Stored list of encoded labels.
        max_length (int): Stored maximum length of the text sequences.
    """
    
    def __init__(self, encoded_texts: list, encoded_labels: list, max_length: int) -> None:
        super().__init__()
        self.encoded_texts = encoded_texts
        self.encoded_labels = encoded_labels
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.encoded_labels)

    def __getitem__(self, idx) -> Tuple[list, int]:
        """sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[list, int]: (encoded_text, label) where encoded_text is the encoded text sequence and label is the corresponding label.
        """
        encoded_text = self.encoded_texts[idx]
        label = self.encoded_labels[idx]
        return encoded_text, label
