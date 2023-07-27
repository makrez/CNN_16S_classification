from Bio import AlignIO
import matplotlib.pyplot as plt
import numpy as np

# Parse the alignment file
alignment = AlignIO.read('alignment.fasta', 'fasta')

# Initialize a figure of appropriate size
plt.figure(figsize=(10, len(alignment)*0.3))

# Create a binary 2D array: 1 for nucleotide match, 0 for mismatch/gap
binary = [[1 if col == alignment[0, i] else 0 for i, col in enumerate(rec)] for rec in alignment]

# Display the binary heatmap
plt.imshow(binary, cmap='hot', interpolation='none', aspect='auto')

# Setting the ticks and labels on the x-axis and y-axis
plt.xticks(np.arange(alignment.get_alignment_length()), list(alignment[0]))
plt.yticks(np.arange(len(alignment)), [rec.id for rec in alignment])

plt.savefig('alignment.png', dpi=300, bbox_inches='tight')
plt.close()

