# Genomic Sequence Classification using NLP: L. donovani vs. T. brucei

An end-to-end machine learning pipeline that applies Natural Language Processing (NLP) techniques to classify DNA sequences. This project specifically focuses on computational parasitology, distinguishing between the protein-coding genomic sequences of two kinetoplastid parasites: *Leishmania donovani* and *Trypanosoma brucei*.

## 🧬 Biological Context & Methodology

Traditional molecular diagnostics often rely on identifying specific biomarkers (like the IST1 gene region) using PCR. This computational approach treats DNA as a "language." By breaking genomic sequences into overlapping substrings of length *k* (k-mers), we can use algorithms originally designed for text classification to identify structural genomic differences between species.

*   **Tokenization:** DNA sequences are split into overlapping 6-mers (e.g., `ATGCGT` -> `ATG TGC GCG CGT`), effectively creating a vocabulary of genomic "words."
*   **Vectorization:** Scikit-learn's `CountVectorizer` transforms these k-mer sentences into a numerical matrix based on word frequency.
*   **Classification:** A Multinomial Naive Bayes model learns the distinct k-mer vocabulary frequencies unique to each pathogen's coding regions.

## 📊 Dataset

The data consists of haploid genome assemblies sourced directly from the National Center for Biotechnology Information (NCBI):
*   *Leishmania donovani* Genomic coding sequences (FASTA)
*   *Trypanosoma brucei* Genomic coding sequences (FASTA)

*(Note: The raw `.fna` files are not included in this repository due to size. You can download them directly from NCBI Datasets).*

## 🚀 Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/chamudithagmck/laboratory_culture_Genome_sequencing.git
   cd laboratory_culture_Genome_sequencing
   ```
2. Install the required dependencies:
   ```bash
   pip install biopython scikit-learn pandas numpy
   ```
3. Download the coding sequence FASTA files from NCBI, rename them to `leishmania_coding.fna` and `trypanosoma_coding.fna`, and place them in the root directory.

## 💻 Usage

Run the classification script from your terminal:
```bash
python dna_nlp.py
```

## 📈 Model Performance & Results

The model achieves exceptional accuracy in distinguishing between the two parasites purely based on k-mer frequencies, demonstrating that these kinetoplastids utilize distinctly different coding "vocabularies."

**Overall Accuracy:** 96.58%

### Confusion Matrix
| | Predicted L. donovani | Predicted T. brucei |
|---|---|---|
| **True L. donovani** | 1522 | 81 |
| **True T. brucei** | 34 | 1723 |

### Classification Report
| Species | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Leishmania donovani for set 1** | 0.98 | 0.95 | 0.96 | 1603 |
| **Trypanosoma brucei for set 1** | 0.96 | 0.98 | 0.97 | 1757 |
| **Leishmania donovani for set 2** | 0.98 | 0.94 | 0.96 | 1617 |
| **Trypanosoma brucei for set 2** | 0.94 | 0.98 | 0.96 | 1754 |

*   **Diagnostic insight:** The model yields high precision (0.98) for *L. donovani*, meaning false positives are exceedingly rare (only 34 misclassified sequences). 

## 🛠️ Tech Stack
*   **Python 3**
*   **Biopython** (Sequence parsing and handling)
*   **Scikit-learn** (Vectorization, Train-Test Split, Naive Bayes Classifier)
*   **NumPy & Pandas** (Data structuring)

---
*Developed by Chamuditha Karunarathna as an exploration into AI applications for molecular biology and bioinformatics.*
