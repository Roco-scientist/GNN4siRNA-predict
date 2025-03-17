import argparse
import pandas as pd
import scipy
import stellargraph
import warnings
import pdb

from Bio import BiopythonDeprecationWarning, SeqIO
from data_processing import get_split_data, get_split_data_gene_model, index_x
from features import siRNA, mRNA
from multiprocessing import Pool, cpu_count
from pathlib import Path
from sklearn.model_selection import train_test_split
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
    return args.parse_args()


def create_generator(thermo_feats_pd, sirna_pd, mrna_pd, y=None, train=False):
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
    if y is None:
        return generator, generator.flow(interactions, shuffle=train)
    else:
        return generator, generator.flow(interactions, y, shuffle=train)


def generate_model(
    x_train, y_train, x_validate, y_validate, x_hold_out=None, y_hold_out=None
):
    # k-mers of siRNA sequences
    sirna_pd_train = pd.DataFrame(data=x_train["siRNA_kmers"])
    sirna_pd_train = sirna_pd_train.set_index(0)
    # k-mers of mRNA sequences
    mRNA_pd_train = pd.DataFrame(data=x_train["mRNA_kmers"])
    mRNA_pd_train = mRNA_pd_train.set_index(0)
    generator, train_gen = create_generator(
        x_train["Thermo_Features"], sirna_pd_train, mRNA_pd_train, y=y_train, train=True
    )

    # k-mers of siRNA sequences
    sirna_pd_validate = pd.DataFrame(data=x_validate["siRNA_kmers"])
    sirna_pd_validate = sirna_pd_validate.set_index(0)
    # k-mers of mRNA sequences
    mRNA_pd_validate = pd.DataFrame(data=x_validate["mRNA_kmers"])
    mRNA_pd_validate = mRNA_pd_validate.set_index(0)
    _, validate_gen = create_generator(
        x_validate["Thermo_Features"], sirna_pd_validate, mRNA_pd_validate, y=y_validate
    )

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

    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=15)
    model.fit(
        train_gen,
        epochs=1000,
        validation_data=validate_gen,
        verbose=2,
        shuffle=False,
        callbacks=[reduce_lr, early_stop],
    )

    if ARGS.evaluate:
        # k-mers of siRNA sequences
        sirna_pd_hold_out = pd.DataFrame(data=x_hold_out["siRNA_kmers"])
        sirna_pd_hold_out = sirna_pd_hold_out.set_index(0)
        # k-mers of mRNA sequences
        mRNA_pd_hold_out = pd.DataFrame(data=x_hold_out["mRNA_kmers"])
        mRNA_pd_hold_out = mRNA_pd_hold_out.set_index(0)
        _, hold_out_gen = create_generator(
            x_hold_out["Thermo_Features"],
            sirna_pd_hold_out,
            mRNA_pd_hold_out,
            y=y_hold_out,
        )

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
    return model


def predict(sirna_pd, mrna_pd, thermo_feats_pd, model):
    _, pred_gen = create_generator(thermo_feats_pd, sirna_pd, mrna_pd)
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
    return pred


def create_thermo_pd_row(sirna_mrna):
    sirna, mrna = sirna_mrna
    return sirna.thermo_features(mrna)


def main():
    thermo_features_file = Path("./data/processed/HUVKS/thermo_features.csv")
    siRNA_antisense_fasta_file = Path("./data/raw/HUVKS/siRNA.fa")
    mRNA_fasta_file = Path("./data/raw/HUVKS/mrna.fa")
    efficacy_file = Path("./data/raw/HUVKS/sirna_mrna_efficacy.csv")
    if ARGS.evaluate:
        x, y, split_indexes, sirna_mrna = get_split_data(
            thermo_features_file,
            siRNA_antisense_fasta_file,
            mRNA_fasta_file,
            efficacy_file,
            folds=6,
        )
        results_file_path = Path("./accuracy.csv")
        with open(results_file_path, "w") as results_file:
            results_file.write(
                "HoldOut,Validation,R2_HoldOut,MSE_HoldOut,PCC_HoldOut,R2_Validation,MSE_Validation,PCC_Validation\n"
            )
        for hold_out_set in range(len(split_indexes)):
            for validation_set in range(len(split_indexes)):
                if hold_out_set != validation_set:
                    hold_out_indexes = split_indexes[hold_out_set]
                    validation_indexes = split_indexes[validation_set]
                    train_indexes = [
                        not hold_out and not validation
                        for hold_out, validation in zip(
                            hold_out_indexes, validation_indexes
                        )
                    ]
                    x_train = index_x(x, train_indexes, sirna_mrna)
                    x_validate = index_x(x, validation_indexes, sirna_mrna)
                    x_hold_out = index_x(x, hold_out_indexes, sirna_mrna)

                    y_train = y[train_indexes]
                    y_validate = y[validation_indexes]
                    y_hold_out = y[hold_out_indexes]

                    r2, mse, pcc, r2_val, mse_val, pcc_val = generate_model(
                        x_train, y_train, x_validate, y_validate, x_hold_out, y_hold_out
                    )
                    with open(results_file_path, "a") as results_file:
                        results_file.write(
                            f"{hold_out_set + 1},{validation_set + 1},{r2},{mse},{pcc},{r2_val},{mse_val},{pcc_val}\n"
                        )
    else:
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
        source_target = pd.read_csv(sirna_mrna_csv_file)

        sequence_info = []
        # create sirna k-mer file
        sirna_records = {}
        sirna_kmers_data = []
        for seq_record in SeqIO.parse(sirna_fasta_file, "fasta"):  # input fasta file
            sirna = siRNA(seq_record)
            sirna_records[sirna.id] = sirna
            sirna_kmers_data.append(sirna.kmer_data())

        mrna_records = {}
        mrna_kmers_data = []
        for seq_record in SeqIO.parse(mrna_fasta_file, "fasta"):
            mrna = mRNA(seq_record)
            mrna_records[mrna.id] = mrna
            mrna_kmers_data.append(mrna.kmer_data())

        sirna_mrna = [
            (sirna_records[sirna_id], mrna_records[mrna_id])
            for sirna_id, mrna_id in zip(
                source_target.iloc[:, 0], source_target.iloc[:, 1]
            )
        ]
        print("\nFinding thermo properties.  May take awhile\n")
        with Pool(ARGS.threads) as pool:
            thermo_rows = pool.map(create_thermo_pd_row, sirna_mrna)

        thermo_feats_pd = pd.DataFrame(thermo_rows)

        x_train, y_train, x_validate, y_validate = get_split_data_gene_model(
            thermo_features_file,
            siRNA_antisense_fasta_file,
            mRNA_fasta_file,
            efficacy_file,
        )
        model = generate_model(x_train, y_train, x_validate, y_validate)
        source_target["Efficacy_Prediction"] = predict(
            pd.DataFrame(data=sirna_kmers_data).set_index(0),
            pd.DataFrame(data=mrna_kmers_data).set_index(0),
            thermo_feats_pd,
            model,
        )

        if not ARGS.test:
            source_target.to_csv(
                sirna_mrna_csv_file.with_suffix(".prediction.csv"), index=False
            )


if __name__ == "__main__":
    ARGS = arguments()
    main()
