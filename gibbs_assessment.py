import numpy as np
import pandas as pd

from features import siRNA, mRNA
from Bio import SeqIO
from tqdm import tqdm


def main():
    sirnas = {
        seqrecord.id: siRNA(seqrecord)
        for seqrecord in SeqIO.parse("./data/raw/dataset_1/sirna_1.fas", "fasta")
    }
    mrnas = {
        seqrecord.id: mRNA(seqrecord)
        for seqrecord in SeqIO.parse("./data/raw/dataset_1/mRNA_1.fas", "fasta")
    }
    sirna_info = {
        row[0]: {"mRNA": row[1], "Gibbs": float(row[-3])}
        for row in np.genfromtxt(
            "./data/processed/dataset_1/sirna_target_thermo.csv",
            delimiter=",",
            dtype=str,
        )
    }

    sirnas_ids = list(sirnas.keys())
    sirnas_ids.sort()

    gibbs_free_energy_values = []  # List of lists, [[sense, antisense, original]]

    for sirna_id in tqdm(sirnas_ids[:100], desc="Calculating Gibbs free energy"):
        sirna = sirnas[sirna_id]
        mrna = mrnas[sirna_info[sirna_id]["mRNA"]]
        if sirna.seq.transcribe() in mrna.seq:  # siRNA strand is sense
            sirna._rnaup(mrna, extension=None, extra=True)
            sense_gibbs = sirna.thermo_row[-3]
            sense_mrna_match = sirna.mrna_sequence_match
            sirna.seq = sirna.seq.reverse_complement_rna()
            sirna._rnaup(mrna, extension=None, extra=True)
            antisense_gibbs = sirna.thermo_row[-3]
            antisense_mrna_match = sirna.mrna_sequence_match
            antisense_sirna_seq = sirna.seq
        elif sirna.seq.reverse_complement() in mrna.seq:  # siRNA strand is antisense
            sirna._rnaup(mrna, extension=None, extra=True)
            antisense_gibbs = sirna.thermo_row[-3]
            antisense_mrna_match = sirna.mrna_sequence_match
            antisense_sirna_seq = sirna.seq
            sirna.seq = sirna.seq.reverse_complement_rna()
            sirna._rnaup(mrna, extension=None, extra=True)
            sense_gibbs = sirna.thermo_row[-3]
            sense_mrna_match = sirna.mrna_sequence_match
        else:
            raise ValueError(
                f"siRNA ({sirna.id}) sequence not found in mRNA ({mrna.id})"
            )
        gibbs_free_energy_values.append(
            (sense_gibbs, antisense_gibbs, sirna_info[sirna_id]["Gibbs"], antisense_sirna_seq, sense_mrna_match, antisense_mrna_match)
        )
    pd.DataFrame(
        data=gibbs_free_energy_values,
        columns=[
            "Sense_siRNA_Gibbs_Free_Energy",
            "Antisense_siRNA_Gibbs_Free_Energy",
            "LaRosa_Gibbs_Free_Energy",
            "Anisense_siRNA_sequence",
            "Sense_siRNA_matching_mRNA_sequence",
            "Antisense_siRNA_matching_mRNA_sequence",
        ],
    ).to_csv("LaRosa_Gibbs.csv", index=False)


if __name__ == "__main__":
    main()
