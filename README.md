# classification_NN_16S

## Dependencies

This repository uses the docker image `docker://makrezdocker/ml-16s:1.0`. Then,
a singularity image is build with:

```
singualriry pull docker://makrezdocker/ml-16s:1.0;
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
thousands of files are loaded during training.

## Input Data

The input data is a multiple-sequence alignment (msa) in `.fasta` format.
The taxonomic assignment of the sequence is parsed from the header. Therefore,
the header has to have the following format:

```
>ncbi_identifier Domain;Kingdom;Phylum;Order;Family;Genus;Species
```

## Data engineering

This part of the repository provides a toolkit to process large DNA alignments.

The basic strategy is to process each sequence in the `msa` and save it as 
a one-hot encoded sequence tensor as a `pt` file. The taxonomy information will 
be saved in a list of dictionaries in a `pkl` file.

Each sequence gets a unique sequence identifier which is used in the filename 
(e.g. `0.pt`), in the `.pt` file and in the taxonomy list.

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

Some of the labels may not contain the required number of taxonomic levels and need 
to be excluded from the data set (if manual curation is impossible).

The script `clean_labels.py` can be used to filter the taxonomy list.

Usage:

```
python clean_labels.py split_dictionary --base_path /path/to/dict/file --dict_file full_labels.pkl
```

#### Retrieve sequence information based on index

It may be useful to retrieve the sequence and taxonomy information given a certain
index.

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

It is often useful and necessary to inspect the number of classes and the counts
before training, which can be done with the script `create_data_statistics.py`.
An example output can be found in `dataset_statistics/`.

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
