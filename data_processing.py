import pandas as pd
import numpy as np
import random

from Bio import SeqIO
from features import siRNA, mRNA
from sklearn.model_selection import KFold
import pdb


def get_input_data(
    thermo_features,
    siRNA_antisense_fasta_file,
    mRNA_fasta_file,
    efficacy_file,
):
    efficacy_data = pd.read_csv(efficacy_file)
    y = efficacy_data.efficacy.to_numpy()
    sirna_mrna = np.array(
        [[sirna, mrna] for sirna, mrna in zip(efficacy_data.siRNA, efficacy_data.mRNA)]
    )

    rnai_antisense_seqs = {
        seq_record.id: seq_record
        for seq_record in SeqIO.parse(siRNA_antisense_fasta_file, "fasta")
    }

    x = {}
    x["Thermo_Features"] = pd.read_csv(thermo_features, header=None)
    x["siRNA_kmers"] = [
        siRNA(rnai_antisense_seqs[rnai_id]).kmer_data() for rnai_id, _ in sirna_mrna
    ]

    x["mRNA_kmers"] = [
        mRNA(mrna_record).kmer_data() for mrna_record in SeqIO.parse(mRNA_fasta_file, "fasta")
    ]
    return (x, y, sirna_mrna)


def get_gene_indexes(folds, source):
    total_samples = len(source)
    gene_occurrances = np.unique(source, return_counts=True)
    proportion_each_gene = {
        gene: occurances / total_samples
        for gene, occurances in zip(gene_occurrances[0], gene_occurrances[1])
    }
    source_left = list(set(source))
    source_left.sort()
    random.seed(42)
    random.shuffle(source_left)
    split_indexes = []
    for _ in range(folds):
        split_proportion = 0
        if len(split_indexes) == folds - 1:
            chosen_genes = source_left
            split_proportion = sum(
                [proportion_each_gene[chosen_gene] for chosen_gene in chosen_genes]
            )
        else:
            chosen_genes = []
            while split_proportion < 1 / folds and len(source_left) != 0:
                # EGFP represents a large proportion of the results, so we start with this gene to prevent
                # One split from being too large
                if "EGFP" in source_left:
                    chosen_gene = "EGFP"
                    source_left.remove("EGFP")
                else:
                    chosen_gene = source_left.pop()
                new_proportion = split_proportion + proportion_each_gene[chosen_gene]
                if new_proportion < 1 / folds * 1.05 or len(chosen_genes) == 0:
                    split_proportion = new_proportion
                    chosen_genes.append(chosen_gene)
                else:
                    source_left.insert(0, chosen_gene)
                    if split_proportion > 1 / folds * 0.95:
                        break
        split_indexes.append([gene in chosen_genes for gene in source])
        print(
            f"Split proportion of {round(split_proportion, 4)} for split {len(split_indexes)}"
        )
    return split_indexes


def get_split_data(
    thermo_features,
    siRNA_antisense_fasta_file,
    mRNA_fasta_file,
    efficacy_file,
    folds,
    by_gene=True,
):
    x, y, sirna_mrna = get_input_data(
        thermo_features,
        siRNA_antisense_fasta_file,
        mRNA_fasta_file,
        efficacy_file,
    )
    source = list(sirna_mrna[:, 1])
    if by_gene:
        split_indexes = get_gene_indexes(folds, source)
    else:
        split_indexes = []
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        index_list = list(range(len(source)))
        for _, validation_index in kf.split(index_list):
            split_indexes.append([x in validation_index for x in index_list])
    return x, y, split_indexes, sirna_mrna


def index_x(x, indexes, sirna_mrna):
    x_subset = {}
    x_subset["Thermo_Features"] = x["Thermo_Features"].loc[indexes]
    x_subset["siRNA_kmers"] = [selected for selected, flag in zip (x["siRNA_kmers"], indexes) if flag]
    train_sirnas = [sirna_data[0] for sirna_data in x_subset["siRNA_kmers"]]
    train_mrnas = [mrna_id for sirna_id, mrna_id in sirna_mrna if sirna_id in train_sirnas]
    x_subset["mRNA_kmers"] = [mrna_data for mrna_data in x["mRNA_kmers"] if mrna_data[0] in train_mrnas]
    return x_subset


def get_split_data_gene_model(
    thermo_features,
    siRNA_antisense_fasta_file,
    mRNA_fasta_file,
    efficacy_file,
):
    x, y, split_indexes, sirna_mrna = get_split_data(
        thermo_features,
        siRNA_antisense_fasta_file,
        mRNA_fasta_file,
        efficacy_file,
        5,
    )
    validation_indexes = split_indexes[3]  # make sure the validation is not GFP
    train_indexes = [not x for x in validation_indexes]
    x_train = index_x(x, train_indexes, sirna_mrna)
    x_validate = index_x(x, validation_indexes, sirna_mrna)
    y_train = y[train_indexes]
    y_validate = y[validation_indexes]
    return x_train, y_train, x_validate, y_validate
