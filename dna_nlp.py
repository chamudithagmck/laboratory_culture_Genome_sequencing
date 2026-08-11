from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer

#1. Define a function to extract k-mers from a DNA sequence
def get_kmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

sequences = []
#2. Load the DNA sequences from a FASTA file
fasta_file = "C:\VS Projects\Leishmania donovani isolate\laboratory_culture_Genome_sequencing\cds_from_genomic.fna"

#3. Generate 6-mers and join them with spaces (like human sentences)
for record in SeqIO.parse(fasta_file, "fasta"):
    seq_str = str(record.seq)
    kmers = get_kmers(seq_str)
    kmers_sentence = ' '.join(kmers)
    sequences.append(kmers_sentence)

print(f"Successfully loaded and tokenized {len(sequences)} sequences.")
print("Example k-mer sentence:", sequences[0][:50], "...")

## 4. Vectorize the 'sentences' into a mathematical matrix
# The vectorizer treats our space-separated k-mers as individual words
cv = CountVectorizer(ngram_range=(1,1))
X = cv.fit_transform(sequences)

print(f"Matrix shape: {X.shape}")
print("Data is now vectorized and ready for a machine learning classifier!")