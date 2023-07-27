## Data engineering

This part of the repository provides a toolkit to process large DNA alignments.

The basic strategy is to process each sequence in the `msa` and save it as 
a sequence tensor as a `pt` file. The taxonomy information will be saved in a 
list of dictionaries in a `pkl` file.

Each sequence gets a unique sequence identifier which is used in the filename 
(e.g. `0.pt`), in the `.pt` file and in the taxonomy list.

Usage:

```
python process_data.py --msa_file_path /path/to/msa.fasta --alignment_length=<int> \
    --dataset_dir /path/to/output/
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





