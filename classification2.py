import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Define function to break DNA into 6-mers
def get_kmers(sequence, size=6):
    return [sequence[i:i+size].lower() for i in range(len(sequence) - size + 1)]

# 2. Load FASTA files and assign labels
def load_genomic_data(file_class_0, file_class_1):
    sequences = []
    labels = []

    # Load Class 0 (e.g., Species A)
    for record in SeqIO.parse(file_class_0, "fasta"):
        kmers = ' '.join(get_kmers(str(record.seq)))
        if len(kmers) > 0:
            sequences.append(kmers)
            labels.append(0)

    # Load Class 1 (e.g., Species B)
    for record in SeqIO.parse(file_class_1, "fasta"):
        kmers = ' '.join(get_kmers(str(record.seq)))
        if len(kmers) > 0:
            sequences.append(kmers)
            labels.append(1)

    return sequences, np.array(labels)

# --- EXECUTION PIPELINE ---

# Replace these with your actual FASTA file paths
file_species_Leishmania_donovani = "leishmania_coding2.fna"
file_species_Trypanosoma_brucei = "trypanosoma_coding2.fna"

print("Loading sequences and creating k-mers...")
sequences, y = load_genomic_data(file_species_Leishmania_donovani, file_species_Trypanosoma_brucei)

# 3. Vectorize k-mers into feature matrix X
print("Vectorizing k-mer vocabulary...")
cv = CountVectorizer(ngram_range=(1, 1))
X = cv.fit_transform(sequences)

# 4. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Initialize and Train Naive Bayes Model
print("Training Multinomial Naive Bayes classifier...")
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# 6. Make Predictions and Evaluate
y_pred = classifier.predict(X_test)

print("\n================ MODEL RESULTS ================")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Leishmania donovani", "Trypanosoma brucei"]))