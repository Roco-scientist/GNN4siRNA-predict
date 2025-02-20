import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import subprocess
import argparse
import stellargraph as StellarGraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import stellargraph as StellarGraph
from stellargraph.mapper import HinSAGENodeGenerator
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, Model, optimizers, callbacks
from stellargraph.layer import HinSAGE
from multiprocessing import Pool, cpu_count
from pathlib import Path
from Bio import BiopythonDeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)


def arguments():
    args = argparse.ArgumentParser(description="Predicts siRNA efficacy")
    args.add_argument("--sirna_fasta", type=str, help="siRNA Fasta file")
    args.add_argument("--mrna_fasta", type=str, help="mRNA Fasta file")
    args.add_argument(
        "--sirna_mrna_csv",
        type=str,
        help="CSV file with siRNA,mRNA columns in order of siRNA Fasta",
    )
    args.add_argument(
        "--threads",
        type=int,
        default=cpu_count(),
        help="Number of threads",
    )
    args.add_argument(
        "--test",
        action="store_true",
        help="Test code",
    )
    return args.parse_args()


class kmer_featurization:
    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ["A", "T", "C", "G"]
        self.multiplyBy = 4 ** np.arange(
            k - 1, -1, -1
        )  # the multiplying number for each digit position in the k-number system
        self.n = 4**k  # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(
        self, seqs, write_number_of_occurrences=False
    ):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        kmer_features = []
        for seq in seqs:
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(
                seq.upper(), write_number_of_occurrences=write_number_of_occurrences
            )
            kmer_features.append(this_kmer_feature)

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(
        self, seq, write_number_of_occurrences=False
    ):
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.

        Args:
          seq:
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n)

        for i in range(number_of_kmers):
            this_kmer = seq[i : (i + self.k)]
            ok = True
            for letter in this_kmer:
                if letter not in self.letters:
                    ok = False
            if ok:
                this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
                kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer):
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:

            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering


def build_kmers(sequence):
    kmers = []
    n_kmers = len(sequence) - 1

    for i in range(n_kmers):
        kmer = sequence[i : i + 2]
        kmers.append(kmer)
    return kmers


def generate_model():
    # k-mers of siRNA sequences
    sirna_kmer_file = "./data/processed/dataset_2/sirna_kmers.txt"
    # k-mers of mRNA sequences
    mrna_kmer_file = "./data/processed/dataset_2/target_kmers.txt"
    # thermodynamic features of siRNA-mRNA interaction
    sirna_target_thermo_file = "./data/processed/dataset_2/sirna_target_thermo.csv"
    # sirna_efficacy_values
    sirna_efficacy_file = "./data/processed/dataset_2/sirna_mrna_efficacy.csv"

    #############################################
    # import file with sirna / target thermodynamic features
    #############################################

    # k-mers of siRNA sequences
    sirna_pd = pd.read_csv(sirna_kmer_file, header=None)
    sirna_pd = sirna_pd.set_index(0)

    # k-mers of mRNA sequences
    mRNA_pd = pd.read_csv(mrna_kmer_file, header=None)
    mRNA_pd = mRNA_pd.set_index(0)

    # thermodynamic features of siRNA-mRNA interaction
    thermo_feats_pd = pd.read_csv(sirna_target_thermo_file, header=None)

    # sirna_efficacy_values
    sirna_efficacy_pd = pd.read_csv(sirna_efficacy_file)

    # rename first 2 columns in "source" and "target"
    thermo_feats_pd.rename(columns={0: "source", 1: "target"}, inplace=True)

    # Here we transform interaction edges in "interaction nodes"
    # Intercation node has 2 edges that connect it to siRNA and mRNA, respectively
    # Node ID cames from source and target ids
    interaction_pd = thermo_feats_pd.drop(["source", "target"], axis=1)
    interaction_pd["index"] = (
        thermo_feats_pd["source"].astype(str) + "_" + thermo_feats_pd["target"]
    )
    interaction_pd = interaction_pd.set_index("index")

    # New edges have no features
    sirna_edge_pd_no_feats = thermo_feats_pd[["source", "target"]]
    data1 = {
        "source": list(interaction_pd.index),
        "target": sirna_edge_pd_no_feats["source"],
    }
    data2 = {
        "source": list(interaction_pd.index),
        "target": sirna_edge_pd_no_feats["target"],
    }

    all_my_edges = pd.DataFrame(data1)
    all_my_edges_temp = pd.DataFrame(data2)

    # Merge all the edges
    all_my_edges = pd.concat(
        [all_my_edges, all_my_edges_temp], ignore_index=True, axis=0
    )

    # We want to predict the interaction weight, i.e. the label of interaction node
    interaction_weight = sirna_efficacy_pd["efficacy"]
    interaction_weight = interaction_weight.set_axis(interaction_pd.index)

    # Create Stellargraph object
    my_stellar_graph = StellarGraph.StellarGraph(
        {"siRNA": sirna_pd, "mRNA": mRNA_pd, "interaction": interaction_pd},
        edges=all_my_edges,
        source_column="source",
        target_column="target",
    )

    ################################################
    # Create the model
    ################################################

    train_interaction, test_interaction = train_test_split(
        interaction_weight, test_size=0.1, random_state=42
    )

    # sizes of 1- and 2-hop neighbour samples for each hidden layer of the HinSAGE model
    batch_size = 60
    hop_samples = [8, 4]
    # Create the generators to feed data from the graph to the Keras model
    # We specify we want to make node regression on the "interaction" node
    generator = HinSAGENodeGenerator(
        my_stellar_graph, batch_size, hop_samples, head_node_type="interaction"
    )

    train_gen = generator.flow(train_interaction.index, train_interaction, shuffle=True)

    # two hidden layers HinSAGE sizes
    hinsage_layer_sizes = [32, 16]
    hinsage_model = HinSAGE(
        layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.15
    )

    # Expose input and output sockets of hinsage:
    x_inp, x_out = hinsage_model.in_out_tensors()

    prediction = layers.Dense(units=1)(x_out)

    # Now let’s create the actual Keras model with the graph inputs x_inp
    # provided by the graph_model and outputs being the predictions
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss="mse")

    # Train the model, keeping track of its loss and accuracy on the training set,
    # and its generalisation performance on the test set

    test_gen = generator.flow(test_interaction.index, test_interaction)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=15)
    # save_best = callbacks.ModelCheckPoint(filepath=, monitor="val_mse", mode="min", save_best_only=True)
    model.fit(
        train_gen,
        epochs=1000,
        validation_data=test_gen,
        verbose=2,
        shuffle=False,
        callbacks=[reduce_lr, early_stop],
    )

    # Now we have trained the model we can evaluate on the test set.
    pred = model.predict(test_gen)

    mse_run = mean_squared_error(test_interaction, pred)
    print("Model MSE: " + str(mse_run))
    return model


def predict(sirna_pd, mRNA_pd, thermo_feats_pd, model):
    # rename first 2 columns in "source" and "target"
    thermo_feats_pd.rename(columns={0: "source", 1: "target"}, inplace=True)
    interaction_pd = thermo_feats_pd.drop(["source", "target"], axis=1)
    interaction_pd["index"] = (
        thermo_feats_pd["source"].astype(str) + "_" + thermo_feats_pd["target"]
    )
    interaction_pd = interaction_pd.set_index("index")
    # New edges have no features
    sirna_edge_pd_no_feats = thermo_feats_pd[["source", "target"]]
    data1 = {
        "source": list(interaction_pd.index),
        "target": sirna_edge_pd_no_feats["source"],
    }
    data2 = {
        "source": list(interaction_pd.index),
        "target": sirna_edge_pd_no_feats["target"],
    }

    all_my_edges = pd.DataFrame(data1)
    all_my_edges_temp = pd.DataFrame(data2)

    # Merge all the edges
    all_my_edges = pd.concat(
        [all_my_edges, all_my_edges_temp], ignore_index=True, axis=0
    )

    batch_size = 60
    hop_samples = [8, 4]
    # Merge all the edges
    all_my_edges = pd.concat(
        [all_my_edges, all_my_edges_temp], ignore_index=True, axis=0
    )
    # Create Stellargraph object
    my_stellar_graph = StellarGraph.StellarGraph(
        {"siRNA": sirna_pd, "mRNA": mRNA_pd, "interaction": interaction_pd},
        edges=all_my_edges,
        source_column="source",
        target_column="target",
    )
    # We specify we want to make node regression on the "interaction" node
    generator = HinSAGENodeGenerator(
        my_stellar_graph, batch_size, hop_samples, head_node_type="interaction"
    )
    test_gen = generator.flow(interaction_pd.index, shuffle=False)
    pred = model.predict(test_gen)
    if ARGS.test:
        print(
            "MSE: "
            + str(
                mean_squared_error(
                    pd.read_csv("./data/raw/test/sirna_mrna_efficacy.csv").efficacy,
                    pred,
                )
            )
        )
    return pred


def find_rnaup(seq):
    couple = seq[3] + "\n" + seq[1]
    # calling RNAup with some specifics
    proc = subprocess.Popen(
        ["RNAup", "-b", "-d2", "--noLP", "-o", "-c", "'S'", "RNAup.out"],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        outs, errs = proc.communicate(input=couple.encode(encoding="utf-8"))
        start = " ("
        end = ")\n"
        s = outs.decode(encoding="utf-8")
        s = s[s.find(start) + len(start) : s.rfind(end)]
        s = s.replace("=", "").replace("+", "")
        s = list(map(float, s.split()))
        return [seq[0], s[0], s[2], s[3]]

    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        return None


def cal_thermo_feature(
    sequence, intermolecular_initiation=4.09, symmetry_correction=0.43
):
    sum_stability = 0
    single_sum = []

    bimers = build_kmers(sequence)

    if sequence[0] == "A":
        sum_stability += 0.45
    if sequence[18] == "U":
        sum_stability += 0.45

    for b in bimers:
        bimer_values = {
            "AA": -0.93,
            "UU": -0.93,
            "AU": -1.10,
            "UA": -1.33,
            "CU": -2.08,
            "AG": -2.08,
            "CA": -2.11,
            "UG": -2.11,
            "GU": -2.24,
            "AC": -2.24,
            "GA": -2.35,
            "UC": -2.35,
            "CG": -2.36,
            "GG": -3.26,
            "CC": -3.26,
            "GC": -3.42,
        }
        stability_value = bimer_values.get(b, 0)
        single_sum.append(stability_value)
        sum_stability += stability_value

    sum_stability += intermolecular_initiation
    sum_stability += symmetry_correction
    single_sum.append(sum_stability)

    return single_sum


def main():
    k_sirna = 3
    k_mrna = 4
    kmer_sirna = kmer_featurization(k_sirna)
    kmer_mrna = kmer_featurization(k_mrna)

    sirna_kmers_data = []
    mrna_kmers_data = []

    total_compute = []
    intermolecular_initiation = 4.09
    simmetry_correction = 0.43

    if ARGS.test:
        mrna_fasta_file = Path("./data/raw/test/mRNA_1.fas")
        sirna_fasta_file = Path("./data/raw/test/sirna_1.fas")
        sirna_mrna_csv_file = Path("./data/raw/test/sirna_mrna_efficacy.csv")
    else:
        if ARGS.mrna_fasta is None:
            raise ValueError("mrna_fasta input required")
        if ARGS.sirna_fasta is None:
            raise ValueError("sirna_fasta input required")
        if ARGS.sirna_mrna_csv is None:
            raise ValueError("sirna_mrna_csv input required")
        mrna_fasta_file = Path(ARGS.mrna_fasta)
        sirna_fasta_file = Path(ARGS.sirna_fasta)
        sirna_mrna_csv_file = Path(ARGS.sirna_mrna_csv)
    records = list(SeqIO.parse(mrna_fasta_file, "fasta"))

    sequence_info = []
    # create sirna k-mer file
    for seq_record in SeqIO.parse(sirna_fasta_file, "fasta"):  # input fasta file
        rna_id = seq_record.id
        seq = seq_record.seq
        seq = seq.upper()
        seq = seq.replace("U", "T")
        k = kmer_sirna.obtain_kmer_feature_for_one_sequence(seq, True)
        k = [int(elem) for elem in k]
        k = [str(elem) for elem in k]
        k.insert(0, str(rna_id))
        sirna_kmers_data.append(k)

        sequence = seq_record.seq
        sequence = sequence[:19]
        seq_id = seq_record.id

        single_sum = cal_thermo_feature(sequence)
        single_sum.insert(0, seq_id)
        total_compute.append(single_sum)

        sir = seq_record.seq.upper().replace("U", "T")
        sir_id = seq_record.id
        rev = Seq.reverse_complement(sir)
        for r in records:
            if rev in r.seq:
                sequence_info.append([sir_id, rev, r.id, r.seq])

    # create mRNA k-mer file
    for seq_record in records:  # input fasta file
        rna_id = seq_record.id
        seq = seq_record.seq
        seq = seq.upper()
        seq = seq.replace("U", "T")
        k = kmer_mrna.obtain_kmer_feature_for_one_sequence(seq, True)
        k = [int(elem) for elem in k]
        k = [str(elem) for elem in k]
        k.insert(0, str(rna_id))
        mrna_kmers_data.append(k)

    # python bash script to calculate all the interactions
    print("Finding RNAup energy.  May take awhile")
    with Pool(ARGS.threads) as pool:
        rnaup_data = pool.map(find_rnaup, sequence_info)

    thermo = pd.DataFrame(total_compute)
    rnaup = pd.DataFrame(rnaup_data)

    thermo_feats_pd = pd.DataFrame()
    source_target = pd.read_csv(sirna_mrna_csv_file, header=0)
    thermo_feats_pd = pd.concat(
        [thermo_feats_pd, source_target.iloc[:, 0:2]],
        axis=1,
    )
    thermo_feats_pd = pd.concat(
        [thermo_feats_pd, thermo.iloc[:, 1:20]],
        axis=1,
    )
    thermo_feats_pd = pd.concat([thermo_feats_pd, rnaup.iloc[:, 1:5]], axis=1)
    thermo_feats_pd.columns = range(thermo_feats_pd.columns.size)

    source_target["Efficacy_Prediction"] = predict(
        pd.DataFrame(data=sirna_kmers_data).set_index(0),
        pd.DataFrame(data=mrna_kmers_data).set_index(0),
        thermo_feats_pd,
        generate_model(),
    )

    if not ARGS.test:
        source_target.to_csv(
            sirna_mrna_csv_file.with_suffix(".prediction.csv"), index=False
        )


if __name__ == "__main__":
    ARGS = arguments()
    main()
