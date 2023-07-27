import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


def encode_data(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    encoded_data = []  # Create an empty list to hold the encoded data

    with torch.no_grad():  # Disable gradient computation
        for batch_data in dataloader:  # Iterate over batches of data
            sequence_data = batch_data['sequence'].permute(0, 2, 1).to(device)  # Swap the second and third dimensions and move data to device

            # Feed the 'sequence' data through the model
            _, mu, _ = model(sequence_data)

            # Add the mean of the latent distribution (mu) to the encoded_data list
            encoded_data.append(mu.cpu().numpy()) 

    # Concatenate the list of arrays and return it
    return np.concatenate(encoded_data)

def inspect_encoded_data(encoded_data):
    print(f"Shape of encoded data: {encoded_data.shape}")
    print(f"First few rows of encoded data:\n{encoded_data[:5]}")
    
    # Compute some basic statistics
    means = encoded_data.mean(axis=0)
    variances = encoded_data.var(axis=0)
    
    print(f"Mean of each latent dimension:\n{means}")
    print(f"Variance of each latent dimension:\n{variances}")

# def perform_tsne(latent_space_samples, save_path):
#     tsne = TSNE(n_components=2, random_state=0)
#     latent_space_tsne = tsne.fit_transform(latent_space_samples)

#     plt.scatter(latent_space_tsne[:, 0], latent_space_tsne[:, 1])
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.title('t-SNE visualization of latent space')

#     plt.savefig(save_path)
#     plt.close()

def perform_tsne(latent_space_samples, labels, taxonomic_level, save_path):
    # Define a mapping of taxonomic levels to their indices
    taxonomic_levels = ['ncbi_identifier', 'Domain', 'Kingdom', 'Phylum', 'Order', 'Family', 'Genus', 'Species']
    taxonomic_level_index = taxonomic_levels.index(taxonomic_level)
    save_path = save_path + taxonomic_level + '.png'

    # Perform t-SNE on your latent space samples
    tsne = TSNE(n_components=2, random_state=0)
    latent_space_tsne = tsne.fit_transform(latent_space_samples)

    # Extract the labels for the taxonomic level of your choice
    color_labels = [label[taxonomic_level_index] for label in labels]

    # Create a list of unique labels and assign a color to each
    unique_labels = list(set(color_labels))
    colors = plt.cm.gist_ncar(np.linspace(0, 1, len(unique_labels)))

    # Create a mapping from label to color
    color_map = dict(zip(unique_labels, colors))

    # Create a scatter plot of your t-SNE transformed data, with points colored by label
    for label in unique_labels:
        idx = [i for i, lbl in enumerate(color_labels) if lbl == label]
        plt.scatter(latent_space_tsne[idx, 0], latent_space_tsne[idx, 1], label=label, color=color_map[label])

    # Add labels and title
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE visualization of latent space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()





def perform_pca(latent_space_samples, save_path):
    pca = PCA(n_components=2)
    latent_space_pca = pca.fit_transform(latent_space_samples)

    plt.scatter(latent_space_pca[:, 0], latent_space_pca[:, 1])
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
    plt.title('PCA visualization of latent space')

    plt.savefig(save_path)
    plt.close()

def get_label(sequence_id, full_labels):
    """
    Returns the full label information for a given sequence_id.

    Parameters:
    sequence_id (str): The sequence_id for which to return the label.
    full_labels (list): The full list of labels.

    Returns:
    list: The full label information for the given sequence_id.
    """
    # Find the entry with the matching sequence_id
    full_label_entry = next((item for item in full_labels if item["sequence_id"] == sequence_id), None)

    # If found, extract the full label
    if full_label_entry is not None:
        full_label = full_label_entry["label"]
    else:
        full_label = None

    return full_label

def select_indices_by_taxonomy(full_labels, taxonomic_level, taxon_value):
    taxonomic_index = {'ncbi_identifier': 0, 'Domain': 1, 'Kingdom': 2, \
                       'Phylum': 3, 'Order': 4, 'Family': 5, \
                        'Genus': 6, 'Species': 7}

    index = taxonomic_index[taxonomic_level]

    # Select sequences that have the given taxonomic label
    selected_ids = {item['sequence_id'] for item in full_labels if item['label'][index] == taxon_value}

    return selected_ids