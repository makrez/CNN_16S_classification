import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

class hot_dna:
    ### Class for One Hot Encoding DNA sequences
    def __init__(self, sequence, taxonomy):
        sequence = sequence.upper()
        self.sequence = self._preprocess_sequence(sequence)
        self.category_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 
                                 '-': 4, 'N': 5}
        if sequence:
            self.onehot = self._onehot_encode(self.sequence)
        # splitting by ';' to get each taxonomy level
        self.taxonomy = taxonomy.split(';') 

    def _preprocess_sequence(self, sequence):
        ambiguous_bases = {'R', 'Y', 'S', 'W', 'K', 'M', 'B', 
                           'D', 'H', 'V', '.',}
        new_sequence = ""
        for base in sequence:
            if base in ambiguous_bases:
                new_sequence += 'N'
            else:
                new_sequence += base
        # replace sequences of four or more '-' characters with 'N' characters
        new_sequence = re.sub('(-{4,})', lambda m: 'N' * len(m.group(1)), 
                              new_sequence)
        return new_sequence

    def _onehot_encode(self, sequence):
        integer_encoded = np.array([self.category_mapping[char] for char in sequence]).reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto', 
                                       handle_unknown='ignore')
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # Fill missing channels with zeros
        full_onehot_encoded = np.zeros((len(sequence), 6))
        full_onehot_encoded[:, :onehot_encoded.shape[1]] = onehot_encoded

        return full_onehot_encoded

    def _onehot_decode(self, onehot_encoded):
        # Reverse the mapping dictionary
        reverse_category_mapping = {v: k for k, v in self.category_mapping.items()}
        # Convert one-hot encoding back to integer encoding
        integer_encoded = np.argmax(onehot_encoded, axis=1)
        # Convert integer encoding back to original sequence
        original_sequence = "".join(reverse_category_mapping[i.item()] for i in integer_encoded)
        return original_sequence

