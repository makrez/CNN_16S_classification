import os
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader
from models import ConvVAE
from data_processing_functions import SequenceDataset
from latent_space_exploration_functions import encode_data, \
    inspect_encoded_data, perform_tsne, select_indices_by_taxonomy
import random

# Set up paths and parameters
data_folder = '/scratch/mk_cas/full_silva_dataset/sequences/'
results_path = 'results/Bacteria_Species_min_8_ConvVAE_bs_32_lr_0.001_ne_200'
output_path = os.path.join(results_path, 'plots')
model_path = os.path.join(results_path, 'final_model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alignment_length = 50000
batch_size = 32
subset_fraction = 0.1 #Fraction of test data to be used
taxonomic_level = 'Phylum'
taxon_value = 'Actinobacteria'

# create output directory
os.makedirs(output_path, exist_ok=True)
# Set output_filenames
tsne_output = 'tsne_plot_' + taxonomic_level + 'Is' + taxon_value + \
    'fraction_' + str(subset_fraction).replace('.','_') + '_'

# Load labels and indices
with open(os.path.join(data_folder, 'full_labels_conform.pkl'), 'rb') as f:
    full_labels = pickle.load(f)

with open(os.path.join(results_path, 'train_indices.pkl'), 'rb') as f:
    test_indices = pickle.load(f)

# Get the sequence IDs that have the specified taxonomic label
selected_ids = select_indices_by_taxonomy(full_labels, taxonomic_level, 
                                          taxon_value)

# Filter test_indices to include only the selected IDs
test_indices = [idx for idx in test_indices \
                if idx['sequence_id'] in selected_ids]

print("Number of test sequences: ", len(test_indices))

subset_size = int(len(test_indices) * subset_fraction)
test_indices = random.sample(test_indices, subset_size)

print("Number of test sequences: ", len(test_indices))

# Convert full_labels to a dictionary for quick lookups
full_labels_dict = {item['sequence_id']: item['label'] for item in full_labels}

# Get labels for the test indices
test_labels = [full_labels_dict.get(str(index['sequence_id']), None) for index in test_indices]

# Load the test dataset
test_dataset = SequenceDataset(data_folder, test_indices)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize and load the model
model = ConvVAE(input_length=alignment_length).to(device)

state_dict = torch.load(model_path)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

# Encode the test data into the latent space
latent_space_samples = encode_data(model, test_dataloader, device)
inspect_encoded_data(latent_space_samples)

# Perform t-SNE and PCA on the encoded data
perform_tsne(latent_space_samples, test_labels, 'Domain', os.path.join(output_path, tsne_output))
perform_tsne(latent_space_samples, test_labels, 'Kingdom', os.path.join(output_path, tsne_output))
perform_tsne(latent_space_samples, test_labels, 'Phylum', os.path.join(output_path, tsne_output))
perform_tsne(latent_space_samples, test_labels, 'Order', os.path.join(output_path, tsne_output))
perform_tsne(latent_space_samples, test_labels, 'Family', os.path.join(output_path, tsne_output))
perform_tsne(latent_space_samples, test_labels, 'Genus', os.path.join(output_path, tsne_output))
perform_tsne(latent_space_samples, test_labels, 'Species', os.path.join(output_path, tsne_output))
