import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import numpy as np
import logging
import json
import pickle
import sys
import yaml
from models import *
from training_functions import train_network 
from data_processing_functions import filter_by_taxonomy, encode_labels, \
    split_data, SequenceDataset, write_counts_to_csv, \
    plot_classification_histogram
from model_evaluation_functions import PlotAndReport

# Load the configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get the data and taxonomy parameters from the config file
data_folder=config['data_folder'] 
alignment_length = config['alignment_length'] 
taxonomic_level= config['taxonomic_level']
taxonomic_group=config['taxonomic_group']
classification_level=config['classification_level']
minimum_samples=config['minimum_samples_per_group']
fraction_of_sequences_to_use=config['fraction_of_sequences_to_use']

# Get the hyperparameters from the config file
lr_list = config['lr']
n_epoch_list = config['n_epoch']
batch_size_list = config['batch_size']
model_list = config['model']

# Define the hyperparameters
param_grid = {
    'lr': lr_list,
    'n_epoch': n_epoch_list,
    'batch_size': batch_size_list,
    'model': model_list
}

grid = ParameterGrid(param_grid)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the data
#-------------------------------------------------------------------------------
# Load full labels
with open(os.path.join(data_folder, 'full_labels_conform.pkl'), 'rb') as f:
    full_labels = pickle.load(f)

filtered_labels, labels_counter, classification_counts, num_classes = filter_by_taxonomy(
    full_labels, 
    taxonomic_level=taxonomic_level,
    taxonomic_group=taxonomic_group, 
    classification_level=classification_level, 
    minimum_samples=minimum_samples,
    fraction=fraction_of_sequences_to_use
    )

# Extracting labels from the filtered_labels
encoded_labels, label_map, le = encode_labels(filtered_labels, classification_level)

# Split the data
train_indices, valid_indices, test_indices = split_data(encoded_labels)

train_dataset = SequenceDataset(data_folder, train_indices)
valid_dataset = SequenceDataset(data_folder, valid_indices)
test_dataset = SequenceDataset(data_folder, test_indices)

criterion = nn.CrossEntropyLoss()

results = []

for params in grid:
    
    # Create a subdirectory for this set of parameters
    directory = f"results/{taxonomic_group}_{classification_level}_min_{minimum_samples}_frac_{fraction_of_sequences_to_use}_{params['model']}_bs_{params['batch_size']}_lr_{params['lr']}_ne_{params['n_epoch']}"

    # Check if the directory already exists
    if os.path.isfile(os.path.join(directory, 'training_finished.txt')):
        continue

    os.makedirs(directory, exist_ok=True)

    # Set up a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(os.path.join(directory, 'training.log'))
    handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    # Write classification counts
    pickle.dump(classification_counts, open(f'{directory}/classification_counts.pkl', 'wb'))
    write_counts_to_csv(classification_counts, f'{directory}/classification_counts.csv')
    #plot_classification_histogram(classification_counts,
    #                             os.path.join(directory, 
    #                                        'classes_histogram.png'))

    # save the label_map
    pickle.dump(label_map, open(f'{directory}/label_map.pkl', 'wb'))

    #save test,train and valid indices
    pickle.dump(train_indices, open(f'{directory}/train_indices.pkl', 'wb'))
    pickle.dump(valid_indices, open(f'{directory}/valid_indices.pkl', 'wb'))
    pickle.dump(test_indices, open(f'{directory}/test_indices.pkl', 'wb'))
    
    # Create a subdirectory for model evaluation
    model_evaluation_dir = os.path.join(directory, 'model_evaluation')
    os.makedirs(model_evaluation_dir, exist_ok=True)
    
    batch_size = params['batch_size']
    model_name = params['model']
    print(model_name)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model_class = globals().get(model_name)
    if model_class is None:
        raise ValueError(f"Invalid model name: {model_name}")
    model = model_class(input_length=alignment_length, num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    logging.info(f"Starting training with lr={params['lr']}, n_epoch={params['n_epoch']}")
    print(f"Training with lr={params['lr']}, n_epoch={params['n_epoch']}")

    train_losses, valid_losses, y_true, y_pred, total_duration = train_network(params['n_epoch'], model, optimizer, criterion, train_dataloader, valid_dataloader, device, directory, logger)
    # Save losses and predictions for later analysis
    np.save(os.path.join(model_evaluation_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(model_evaluation_dir, 'valid_losses.npy'), np.array(valid_losses))
    np.save(os.path.join(model_evaluation_dir, 'y_true.npy'), np.array(y_true))
    np.save(os.path.join(model_evaluation_dir, 'y_pred.npy'), np.array(y_pred))

    # Log the results
    logging.info(f"lr={params['lr']}, n_epoch={params['n_epoch']}, train_loss={train_losses[-1]}, valid_loss={valid_losses[-1]}")

    # Save results
    results.append({
        'lr': params['lr'],
        'n_epoch': params['n_epoch'],
        'train_loss': train_losses[-1],  # Final training loss
        'valid_loss': valid_losses[-1],  # Final valid loss
        'model_architecture' : str(model),
        'batch_size': batch_size,
    })

    # Save the results to a JSON file in the model_evaluation subdirectory
    with open(os.path.join(model_evaluation_dir, 'results.json'), 'w') as f:
        json.dump(results[-1], f)
    torch.save(model.state_dict(), os.path.join(directory, 'model_weights.pt'))

    # Create plots and F1 statistics
    plot_and_report = PlotAndReport()
    plot_and_report.plot_confusion_matrix(y_true, y_pred, label_map, model_evaluation_dir)
    plot_and_report.plot_train_test_curves(os.path.join(model_evaluation_dir, 'train_losses.npy'), 
                                           os.path.join(model_evaluation_dir,  'valid_losses.npy'), 
                                           model_evaluation_dir)
    plot_and_report.print_f1_and_classification_report(y_true, y_pred, label_map, model_evaluation_dir)
    with open(os.path.join(directory, 'training_finished.txt'), 'w'):
        print(f"Total training time: {total_duration:.2f}s")
