from Bio import SeqIO
from collections import Counter

def analyze_fasta(filename):
    bases_counter = Counter()
    total_bases = 0

    # Read the fasta file and count the nucleotides and gaps
    for record in SeqIO.parse(filename, "fasta"):
        sequence = str(record.seq)
        bases_counter.update(sequence)
        total_bases += len(sequence)

    print("Total bases (including gaps):", total_bases)

    # Print out the counts and percentages
    for base, count in bases_counter.items():
        print("Count of base", base, ":", count)
        print("Percentage of base", base, ":", (count/total_bases)*100, "%")

analyze_fasta("../../data/subset_bacteria_SILVA_138.1_SSURef_tax_silva_full_align_trunc.fasta")