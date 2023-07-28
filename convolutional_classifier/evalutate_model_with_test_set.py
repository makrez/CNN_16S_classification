import os
import pickle
import torch
from torch.utils.data import DataLoader
from models import LargerModel, SmallModel, ConvClassifier2, \
    ModelWithDropout
from data_processing_functions import SequenceDataset
import json
from model_evaluation_functions import PlotAndReport
import torch.nn as nn
import torch.nn.functional as F

# Set up paths and parameters
data_folder = '/scratch/mk_cas/full_silva_dataset/sequences/'
results_path = 'results/Actinobacteria_Genus_min_20_ConvClassifier2_bs_32_lr_0.001_ne_50'
output_path = os.path.join(results_path, 'final_evaluation')
model_path = os.path.join(results_path, 'final_model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alignment_length = 50000
batch_size = 64
num_classes = 94
model = ConvClassifier2(input_length=alignment_length, num_classes=num_classes).to(device)

# create output directory
os.makedirs(output_path, exist_ok=True)

# Load labels and indices
with open(os.path.join(data_folder, 'full_labels_conform.pkl'), 'rb') as f:
    full_labels = pickle.load(f)

with open(os.path.join(results_path, 'train_indices.pkl'), 'rb') as f:
    test_indices = pickle.load(f)

# Load the test dataset
test_dataset = SequenceDataset(data_folder, test_indices)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

state_dict = torch.load(model_path)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load the new state dict into the model
model.load_state_dict(new_state_dict)


# test phase
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
test_losses = []
y_true = []
y_pred = []
model.eval()

params = {
    "criterion": str(criterion),
    "test_dataloader_length": len(test_dataloader.dataset),
    "device": str(device),
}

with torch.no_grad():
    for i, batch in enumerate(test_dataloader, 1):
        sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
        labels = batch["label"].to(device)
    
        outputs = model(sequence_data)
        _, preds = torch.max(outputs, 1)

        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())

        loss = criterion(outputs, labels)
        test_loss += loss.item() * sequence_data.size(0)

test_loss /= len(test_dataloader.dataset)
test_losses.append(test_loss)

correct_preds = torch.eq(torch.max(F.softmax(outputs, dim=-1), dim=-1)[1], labels).float().sum()
params['correct_preds'] = correct_preds.item()
total_preds = torch.FloatTensor([labels.size(0)])
params['total_preds'] = total_preds.item()
correct_preds = correct_preds.to(device)
total_preds = total_preds.to(device)
accuracy = correct_preds / total_preds
params['accuracy'] = accuracy.item()

with open(f"{output_path}/parameters.json", 'w') as f:
    json.dump(params, f)

# Load labels map
with open(os.path.join(results_path, 'label_map.pkl'), 'rb') as f:
    label_map = pickle.load(f)

# Create plots and F1 statistics
plot_and_report = PlotAndReport()
plot_and_report.plot_confusion_matrix(y_true, y_pred, label_map, output_path)
plot_and_report.print_f1_and_classification_report(y_true, y_pred, label_map, output_path)