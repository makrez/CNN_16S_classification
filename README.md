---
editor_options: 
  markdown: 
    wrap: 72
---

# Convolutional Neural Networds for the lassification of 16S rRNA sequences

## Dependencies

This repository uses the docker image
`docker://makrezdocker/ml-16s:1.0`. Then, a singularity image is build
with:

```         
singualriry pull docker://makrezdocker/ml-16s:1.2;
```

For running the container, log in to a node with GPU support:

```         
srun --pty  --nodelist <node> -c 2 --mem=4000 --time=02:00:00 /bin/bash
```

Test if GPU is available:

```         
singularity exec --nv --bind /data ../singularity_images/ml-16s_1.0.sif python ./check_gpu.py
```

**IMPORTANT**

It is recommended to use the `scratch` partition, since potentially
thousands of files are being read and written during data processing and
training.

## Input Data

The input data is a multiple-sequence alignment (msa) in `.fasta`
format. The taxonomic assignment of the sequence is parsed from the
header. Therefore, the header has to have the following format:

```         
>ncbi_identifier Domain;Kingdom;Phylum;Order;Family;Genus;Species
```

## Data engineering

This part of the repository provides a toolkit to process large DNA
alignments.

The basic strategy is to process each sequence in the `msa` and save it
as a one-hot encoded sequence tensor as a `pt` file. The taxonomy
information will be saved in a list of dictionaries in a `pkl` file.

Each sequence gets a unique sequence identifier which is used in the
filename (e.g. `0.pt`), in the `.pt` file and in the taxonomy list.

Usage:

```         
python process_data.py --msa_file_path /path/to/msa.fasta --alignment_length=<int> \
    --dataset_dir /path/to/output/
```

The output in `/path/to/output/sequences` conatains the `pt` files and
the `pkl` label file:

```         
├── 1.pt
├── 2.pt
├── 3.pt
├── 4.pt
├── 5.pt
└── full_labels.pkl
```

### Utilities

#### Clean labels

Some of the labels may not contain the required number of taxonomic
levels and need to be excluded from the data set (if manual curation is
impossible).

The script `clean_labels.py` can be used to filter the taxonomy list.

Usage:

```         
python clean_labels.py split_dictionary --base_path /path/to/dict/file --dict_file full_labels.pkl
```

This function outputs the files `full_labels_conform.pkl` and
`full_labels_non_conform.pkl`.

#### Retrieve sequence information based on index

It may be useful to retrieve the sequence and taxonomy information given
a certain index.

Usage:

```         
python retrieve_sequence.py --sequence_id 0 --sequence_path /path/to/ptfiles/ --msa_file_path /path/to/msa.fasta
```

Example output:

```         
Tensor: tensor([[0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1.],
        ...,
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1.]])
Full label: ['AY855839.1.1390', 'Bacteria', 'Proteobacteria', 'Alphaproteobacteria', 'Rickettsiales', 'Mitochondria', 'Maytenus hookeri']
Original FASTA header: AY855839.1.1390 Bacteria;Proteobacteria;Alphaproteobacteria;Rickettsiales;Mitochondria;Maytenus hookeri
```

#### Create dataset statistics

It is often useful and necessary to inspect the number of classes and
the counts before training, which can be done with the script
`create_data_statistics.py`. An example output can be found in
`dataset_statistics/`.

Usage:

```         
python create_data_statistics.py --labels_file full_labels_conform.pkl \
    --taxonomic_level Family \
    --taxonomic_group Microbacteriaceae \
    --classification_level Genus \
    --minimum_samples 100 \
    --fraction 1 \
    --output_path dataset_statistics/ \
    --make_plot=True
```

## Convolutional Classifier

This part of the code base provides the code for training a
convolutional neural network to classify sequences in the alignment.

### Configuration

The configuration can be set with the `config.yaml` file:

```{yaml}
# Data parameters
data_folder: "/scratch/mk_cas/full_silva_dataset/sequences/"
alignment_length: 50000

# Taxonomy parameters
taxonomic_level: "Phylum"
taxonomic_group: "Actinobacteria"
classification_level: "Genus"
minimum_samples_per_group: 20
fraction_of_sequences_to_use: 0.2

# Hyperparameters
lr: [0.001, 0.0001]
n_epoch: [30]
batch_size: [32]
model: [ConvClassifier2, SmallModel, ModelWithDropout, LargerModel]
```

-   data_folder: The folder where the `.pt` files and the
    `full_labels.pkl` are located. This will be the same folder as the
    `--dataset_dir` in the processing step

-   alignment_length: length of the alignment

-   taxonomic_level: Which taxonomic level should be included ("outer
    level")

-   taxonomic_group: A string that filters for a taxonomic name in
    `taxonomic_level`. In the example above, only sequences from Phylum
    Actinobacteria will be used to create the dataset.

-   classification_level: The choice of which taxonomic level will be
    the classificatoin level. Possible choices are 'Kingdom', 'Phylum',
    'Order', 'Family', 'Genus' & 'Species'.

-   minimu_samples_per_group: The minimum amount of sequences that have
    to present in a class for the class to be considered in the dataset

-   fraction_of_sequences: The fraction of sequences to be included.
    Note that the `minimum_samples_per_group` is applied after
    subsetting randomly for a fraction to use.

-   lr: list of learning learning rates

-   n_epoch: number of epochs (single integer)

-   batch_size: list of batch sizes to be considered

-   model: list of models to train.

Usage:

```{bash, eval=FALSE}
python train.py
```

### Output Files

The script produces the folder `results`. Within this folder, subfolders
are created dynamically for each training combination. Training with the
config variables described above will create the following subfolders:

``` bash
Actinobacteria_Genus_min_20_frac_0.2_ConvClassifier2_bs_32_lr_0.0001_ne_30
Actinobacteria_Genus_min_20_frac_0.2_ConvClassifier2_bs_32_lr_0.001_ne_30
Actinobacteria_Genus_min_20_frac_0.2_LargerModel_bs_32_lr_0.0001_ne_30
Actinobacteria_Genus_min_20_frac_0.2_LargerModel_bs_32_lr_0.001_ne_30
Actinobacteria_Genus_min_20_frac_0.2_ModelWithDropout_bs_32_lr_0.0001_ne_30
Actinobacteria_Genus_min_20_frac_0.2_ModelWithDropout_bs_32_lr_0.001_ne_30
Actinobacteria_Genus_min_20_frac_0.2_SmallModel_bs_32_lr_0.0001_ne_30
Actinobacteria_Genus_min_20_frac_0.2_SmallModel_bs_32_lr_0.001_ne_30
```

Within these subfolders, a subfolder `model_evaluation` is created with
the confusion matrix, F1 classification report. Also, the indices for
train, test and validation set are stored, together with the model
weights.

### Utility scripts

Once the training is finished, and a model is chosen based on the
performance measured with the validation set, the script
`evalutate_model_with_test_set.py` facilitates to easily calculate the
necessary statistics, given the test set (which is automatically stored
in the results subfolder).

Usage:

Adjust the parameters in the script:

``` python
# Set up paths and parameters
data_folder = '/scratch/mk_cas/full_silva_dataset/sequences/'
results_path = 'results/Actinobacteria_Genus_min_20_frac_0.2_ConvClassifier2_bs_32_lr_0.0001_ne_30'
output_path = os.path.join(results_path, 'final_evaluation')    
model_path = os.path.join(results_path, 'final_model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alignment_length = 50000
batch_size = 32
num_classes = 95
model = ConvClassifier2(input_length=alignment_length, num_classes=num_classes).to(device)
```

Then:

```         
python evalutate_model_with_test_set.py
```

This will create a subfolder called `final_evaluation` where the F1
classification report and the confusion matrix can be found.

## Variational Autoencoder

The structure of the training procedure is the same as above. The
`config.yaml` has to be adjusted, followed by `python train.py`.

### Utility scripts

After training, the exploration of the latent space can be achieved with
the file can be achieved with the script
`latent_space_exploration_functions.py`. Adjust the following lines:

``` python
# Set up paths and parameters
data_folder = '/scratch/mk_cas/full_silva_dataset/sequences/'
results_path = 'results/Bacteria_Species_min_8_ConvVAE_bs_32_lr_0.001_ne_200'
output_path = os.path.join(results_path, 'plots')
model_path = os.path.join(results_path, 'final_model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alignment_length = 50000
batch_size = 32
subset_fraction = 0.1 #Fraction of test data to be used
taxonomic_level = 'Domain'
taxon_value = 'Bacteria
```

The output will create tSNE plots in the `results/<subfolder>/plots`
directory. For each taxonomic level, a plot is generated.
