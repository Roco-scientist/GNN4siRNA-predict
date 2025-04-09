import argparse
import numpy as np
import pandas as pd
import scipy
import stellargraph
import tensorflow as tf
import warnings
import pdb

from Bio import BiopythonDeprecationWarning, SeqIO
from data_processing import get_split_data, get_split_data_gene_model, index_x
from features import siRNA, mRNA
from multiprocessing import Pool, cpu_count
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE
from tensorflow.keras import layers, Model, optimizers, callbacks

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
    args.add_argument(
        "--evaluate",
        action="store_true",
        help="Test code",
    )
    args.add_argument(
        "--random_split",
        action="store_true",
        help="Split hold out and validation sets randomly instead of by gene",
    )
    return args.parse_args()


def create_generator(thermo_feats_pd, sirna_pd, mrna_pd):
    thermo_feats_pd.rename(columns={0: "source", 1: "target"}, inplace=True)
    interaction_pd = thermo_feats_pd.drop(["source", "target"], axis=1)
    interactions = list(
        thermo_feats_pd["source"].astype(str) + "_" + thermo_feats_pd["target"]
    )
    interaction_pd["index"] = interactions
    interaction_pd = interaction_pd.set_index("index")
    # New edges have no features
    sirna_edge_pd_no_feats = thermo_feats_pd[["source", "target"]]
    data1 = {
        "source": interactions,
        "target": sirna_edge_pd_no_feats["source"],
    }
    data2 = {
        "source": interactions,
        "target": sirna_edge_pd_no_feats["target"],
    }

    all_my_edges = pd.DataFrame(data1)
    all_my_edges_temp = pd.DataFrame(data2)

    # Merge all the edges
    all_my_edges = pd.concat(
        [all_my_edges, all_my_edges_temp], ignore_index=True, axis=0
    )

    # sizes of 1- and 2-hop neighbour samples for each hidden layer of the HinSAGE model
    batch_size = 60
    hop_samples = [8, 4]
    # Merge all the edges
    all_my_edges = pd.concat(
        [all_my_edges, all_my_edges_temp], ignore_index=True, axis=0
    )
    # Create Stellargraph object
    my_stellar_graph = stellargraph.StellarGraph(
        {"siRNA": sirna_pd, "mRNA": mrna_pd, "interaction": interaction_pd},
        edges=all_my_edges,
        source_column="source",
        target_column="target",
    )
    # We specify we want to make node regression on the "interaction" node
    generator = HinSAGENodeGenerator(
        my_stellar_graph, batch_size, hop_samples, head_node_type="interaction"
    )
    return generator


def generate_model(
    x_train,
    y_train,
    x_validate,
    y_validate,
    x_hold_out=None,
    y_hold_out=None,
    x_predict=None,
):
    """
    Generates the GNN model and predicts on x_hold_out or x_predict.  For the former it is during
    evaluation mode.
    """
    # k-mers of siRNA sequences
    sirna_kmer_data = x_train["siRNA_kmers"]
    sirna_kmer_data.extend(x_validate["siRNA_kmers"])

    mrna_kmer_data = x_train["mRNA_kmers"]
    mrna_kmer_data.extend(x_validate["mRNA_kmers"])

    thermo_pd = pd.concat(
        [x_train["Thermo_Features"], x_validate["Thermo_Features"]],
        ignore_index=True,
        sort=False,
    )

    # If we are doing hold-out cross validation, add these values for the genorator
    if x_hold_out is not None:
        sirna_kmer_data.extend(x_hold_out["siRNA_kmers"])
        mrna_kmer_data.extend(x_hold_out["mRNA_kmers"])
        thermo_pd = pd.concat(
            [thermo_pd, x_hold_out["Thermo_Features"]], ignore_index=True, sort=False
        )

    if x_predict is not None:
        sirna_kmer_data.extend(x_predict["siRNA_kmers"])
        mrna_kmer_data.extend(x_predict["mRNA_kmers"])
        thermo_pd = pd.concat(
            [thermo_pd, x_predict["Thermo_Features"]], ignore_index=True, sort=False
        )

    sirna_pd = pd.DataFrame(data=sirna_kmer_data).drop_duplicates(subset=[0])
    sirna_pd = sirna_pd.set_index(0)
    # k-mers of mRNA sequences
    mRNA_pd = pd.DataFrame(data=mrna_kmer_data).drop_duplicates(subset=[0])
    mRNA_pd = mRNA_pd.set_index(0)
    generator = create_generator(thermo_pd, sirna_pd, mRNA_pd)

    interactions_train = list(
        x_train["Thermo_Features"].iloc[:, 0].astype(str)
        + "_"
        + x_train["Thermo_Features"].iloc[:, 1]
    )
    train_gen = generator.flow(interactions_train, y_train, shuffle=True)

    interactions_validate = list(
        x_validate["Thermo_Features"].iloc[:, 0].astype(str)
        + "_"
        + x_validate["Thermo_Features"].iloc[:, 1]
    )
    validate_gen = generator.flow(interactions_validate, y_validate, shuffle=False)

    ################################################
    # Create the model
    ################################################

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

    reduce_lr = ReduceLROnPlateauWithBestWeights(
        monitor="val_loss", factor=0.5, patience=10
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    model.fit(
        train_gen,
        epochs=1000,
        validation_data=validate_gen,
        verbose=2,
        shuffle=False,
        callbacks=[reduce_lr, early_stop],
    )

    if ARGS.evaluate:
        interactions_hold_out = list(
            x_hold_out["Thermo_Features"].iloc[:, 0].astype(str)
            + "_"
            + x_hold_out["Thermo_Features"].iloc[:, 1]
        )
        hold_out_gen = generator.flow(interactions_hold_out, shuffle=False)

        pred = model.predict(hold_out_gen).flatten()

        r2 = r2_score(y_hold_out, pred)
        mse = mean_squared_error(y_hold_out, pred)
        pearson = scipy.stats.pearsonr(y_hold_out, pred)
        pcc = pearson[0]

        pred = model.predict(validate_gen).flatten()
        r2_val = r2_score(y_validate, pred)
        mse_val = mean_squared_error(y_validate, pred)
        pearson_val = scipy.stats.pearsonr(y_validate, pred)
        pcc_val = pearson_val[0]

        print("Model R^2: " + str(r2))
        print("Model MSE: " + str(mse))
        print("Model PCC: " + str(pcc))
        return r2, mse, pcc, r2_val, mse_val, pcc_val
    if x_predict is not None:
        interactions_predict = list(
            x_predict["Thermo_Features"].iloc[:, 0].astype(str)
            + "_"
            + x_predict["Thermo_Features"].iloc[:, 1]
        )
        pred_gen = generator.flow(interactions_predict, shuffle=False)
        pred = model.predict(pred_gen)
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
        return [round(prediction, 5) for prediction in pred.flatten()]
    return model


class ReduceLROnPlateauWithBestWeights(callbacks.ReduceLROnPlateau):
    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        restore_best_weights=True,
    ):
        super().__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
        )
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_loss = np.inf if self.mode == "min" else -np.inf

    def on_train_begin(self, logs=None):
        self.best_weights = None
        self.best_loss = np.inf if self.mode == "min" else -np.inf
        super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)

        if current_loss is None:
            return

        # Save best weights
        if (self.mode == "min" and current_loss < self.best_loss - self.min_delta) or (
            self.mode == "max" and current_loss > self.best_loss + self.min_delta
        ):
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()

        old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        super().on_epoch_end(epoch, logs)
        new_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # Restore best weights if LR is reduced
        if (
            self.restore_best_weights
            and new_lr < old_lr
            and self.best_weights is not None
        ):
            if self.verbose > 0:
                print("Restoring best weights from epoch with lowest monitored loss")
            self.model.set_weights(self.best_weights)


def create_thermo_pd_row(sirna_mrna):
    sirna, mrna = sirna_mrna
    return sirna.thermo_features(mrna)


def evaluate_model(
    thermo_features_file, siRNA_antisense_fasta_file, mRNA_fasta_file, efficacy_file
):
    x, y, split_indexes, sirna_mrna = get_split_data(
        thermo_features_file,
        siRNA_antisense_fasta_file,
        mRNA_fasta_file,
        efficacy_file,
        folds=6,
        by_gene=not ARGS.random_split,
    )
    if ARGS.random_split:
        results_file_path = Path("./accuracy.random.csv")
    else:
        results_file_path = Path("./accuracy.gene.csv")

    with open(results_file_path, "w") as results_file:
        results_file.write(
            "HoldOut,Validation,R2_HoldOut,MSE_HoldOut,PCC_HoldOut,R2_Validation,MSE_Validation,PCC_Validation\n"
        )

    for hold_out_set, hold_out_indexes in enumerate(split_indexes):
        for validation_set, validation_indexes in enumerate(split_indexes):
            if hold_out_set != validation_set:
                train_indexes = [
                    not hold_out and not validation
                    for hold_out, validation in zip(
                        hold_out_indexes, validation_indexes
                    )
                ]
                x_train, x_validate, x_hold_out = (
                    index_x(x, train_indexes, sirna_mrna),
                    index_x(x, validation_indexes, sirna_mrna),
                    index_x(x, hold_out_indexes, sirna_mrna),
                )
                y_train, y_validate, y_hold_out = (
                    y[train_indexes],
                    y[validation_indexes],
                    y[hold_out_indexes],
                )

                r2, mse, pcc, r2_val, mse_val, pcc_val = generate_model(
                    x_train, y_train, x_validate, y_validate, x_hold_out, y_hold_out
                )

                with open(results_file_path, "a") as results_file:
                    results_file.write(
                        f"{hold_out_set + 1},{validation_set + 1},{r2},{mse},{pcc},{r2_val},{mse_val},{pcc_val}\n"
                    )


def prepare_prediction_data():
    if ARGS.test:
        return (
            Path("./data/raw/test/mRNA_1.fas"),
            Path("./data/raw/test/sirna_1.fas"),
            Path("./data/raw/test/sirna_mrna_efficacy.csv"),
        )
    if not ARGS.mrna_fasta or not ARGS.sirna_fasta or not ARGS.sirna_mrna_csv:
        raise ValueError(
            "mrna_fasta, sirna_fasta, and sirna_mrna_csv inputs are required"
        )
    return Path(ARGS.mrna_fasta), Path(ARGS.sirna_fasta), Path(ARGS.sirna_mrna_csv)


def load_sequence_data(sirna_fasta_file, mrna_fasta_file):
    sirna_records, sirna_kmers_data = {}, []
    for seq_record in SeqIO.parse(sirna_fasta_file, "fasta"):
        sirna = siRNA(seq_record)
        sirna_records[sirna.id] = sirna
        sirna_kmers_data.append(sirna.kmer_data())

    mrna_records, mrna_kmers_data = {}, []
    for seq_record in SeqIO.parse(mrna_fasta_file, "fasta"):
        mrna = mRNA(seq_record)
        mrna_records[mrna.id] = mrna
        mrna_kmers_data.append(mrna.kmer_data())

    return sirna_records, sirna_kmers_data, mrna_records, mrna_kmers_data


def predict_efficacy(
    thermo_features_file, siRNA_antisense_fasta_file, mRNA_fasta_file, efficacy_file
):
    mrna_fasta_file, sirna_fasta_file, sirna_mrna_csv_file = prepare_prediction_data()
    source_target = pd.read_csv(sirna_mrna_csv_file)

    sirna_records, sirna_kmers_data, mrna_records, mrna_kmers_data = load_sequence_data(
        sirna_fasta_file, mrna_fasta_file
    )

    sirna_mrna = [
        (sirna_records[sirna_id], mrna_records[mrna_id])
        for sirna_id, mrna_id in zip(source_target.iloc[:, 0], source_target.iloc[:, 1])
    ]

    print("Finding thermo properties. This may take a while")
    with Pool(ARGS.threads) as pool:
        thermo_rows = pool.map(create_thermo_pd_row, sirna_mrna)

    x_train, y_train, x_validate, y_validate = get_split_data_gene_model(
        thermo_features_file, siRNA_antisense_fasta_file, mRNA_fasta_file, efficacy_file
    )

    x_predict = {
        "Thermo_Features": pd.DataFrame(thermo_rows),
        "siRNA_kmers": sirna_kmers_data,
        "mRNA_kmers": mrna_kmers_data,
    }

    source_target["Efficacy_Prediction"] = generate_model(
        x_train, y_train, x_validate, y_validate, x_predict=x_predict
    )

    if not ARGS.test:
        source_target.to_csv(
            sirna_mrna_csv_file.with_suffix(".prediction.csv"), index=False
        )


def main():
    thermo_features_file = Path("./data/processed/HUVKS/thermo_features.csv")
    siRNA_antisense_fasta_file = Path("./data/raw/HUVKS/siRNA.fa")
    mRNA_fasta_file = Path("./data/raw/HUVKS/mrna.fa")
    efficacy_file = Path("./data/raw/HUVKS/sirna_mrna_efficacy.csv")

    if ARGS.evaluate:
        evaluate_model(
            thermo_features_file,
            siRNA_antisense_fasta_file,
            mRNA_fasta_file,
            efficacy_file,
        )
    else:
        predict_efficacy(
            thermo_features_file,
            siRNA_antisense_fasta_file,
            mRNA_fasta_file,
            efficacy_file,
        )


if __name__ == "__main__":
    ARGS = arguments()
    main()
