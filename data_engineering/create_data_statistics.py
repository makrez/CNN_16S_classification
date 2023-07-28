import random
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import csv
import os
import fire

def filter_by_taxonomy(full_labels, taxonomic_level, taxonomic_group, classification_level, minimum_samples, fraction):
    taxonomic_levels = ['ncbi_identifier', 'Domain', 'Kingdom', 'Phylum', 'Order', 'Family', 'Genus', 'Species']
    filter_words = ['uncultured', 'unidentified', 'metagenome', 'bacterium']

    assert taxonomic_level in taxonomic_levels, f"Invalid taxonomic_level. Must be one of {taxonomic_levels}"
    assert classification_level in taxonomic_levels, f"Invalid classification_level. Must be one of {taxonomic_levels}"
    
    level_index = taxonomic_levels.index(taxonomic_level)
    classification_level_index = taxonomic_levels.index(classification_level)
    filtered_labels = [label for label in full_labels if label['label'][level_index] == taxonomic_group and 
                       not any(word in label['label'][classification_level_index] for word in filter_words) and 
                       not (label['label'][classification_level_index].endswith('sp.') and 'subsp.' not in label['label'][classification_level_index])]
    
    random.shuffle(filtered_labels)
    
    filtered_labels = filtered_labels[:int(fraction * len(filtered_labels))]
    
    classification_counts = Counter([label['label'][classification_level_index] for label in filtered_labels if Counter(
        [label['label'][classification_level_index] for label in filtered_labels])[label['label'][classification_level_index]] >= minimum_samples])
    
    return classification_counts

def plot_classification_histogram(classification_counts, save_path=None, 
                                  classification_level='', taxonomic_level='', 
                                  taxonomic_group=''):
    classification_levels, counts = zip(*sorted(classification_counts.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(max(len(classification_levels) * 0.5, 6), 6))
    plt.subplots_adjust(bottom=0.4)

    ax.bar(range(len(classification_levels)), counts)
    ax.set_xticks(range(len(classification_levels)))
    ax.set_xticklabels(classification_levels, rotation=90, ha='right')
    plt.xlabel(classification_level)
    plt.ylabel('Count')
    plt.title(f'{taxonomic_level} {taxonomic_group}')

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def main(labels_file, taxonomic_level, taxonomic_group, classification_level, 
         minimum_samples, fraction, output_path, make_plot=False):
    base_out = '_'.join([taxonomic_level, taxonomic_group,classification_level,
                          str(minimum_samples),str(fraction)])
    out_csv = os.path.join(output_path, base_out + ".csv")
    out_png = os.path.join(output_path, base_out + ".png")

    with open(labels_file, 'rb') as f:
        full_labels = pickle.load(f)

    classification_counts = filter_by_taxonomy(full_labels, taxonomic_level, 
                                               taxonomic_group,
                                                classification_level, 
                                                minimum_samples, fraction)
    
    with open(out_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Classification Level", "Count"])
        for key, count in classification_counts.items():
            writer.writerow([key, count])

    if make_plot:
        plot_classification_histogram(classification_counts, out_png)
        plot_classification_histogram(classification_counts, out_png, 
                                      classification_level, taxonomic_level, 
                                      taxonomic_group)
if __name__ == '__main__':
    fire.Fire(main)
