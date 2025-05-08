import numpy as np
import subprocess
import re
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class kmer_featurization:
    def __init__(self, k):
        """
        Initialize k-mer featurization.

        Args:
            k (int): The k-mer length.
        """
        self.k = k
        self.letters = {"A": 0, "T": 1, "U": 1, "C": 2, "G": 3}
        self.multiplyBy = 4 ** np.arange(
            k - 1, -1, -1
        )  # the multiplying number for each digit position in the k-number system
        self.n = 4**k  # number of possible k-mers
        self.seq = None

    def obtain_kmer_feature_for_one_sequence(self, write_number_of_occurrences=False):
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.

        Args:
          seq:
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(self.seq) - self.k + 1

        kmer_feature = np.zeros(self.n)

        for i in range(number_of_kmers):
            this_kmer = self.seq[i : (i + self.k)]
            ok = True
            for letter in this_kmer:
                if letter not in self.letters.keys():
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

            digits.append(self.letters[letter])

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering

    def kmer_data(self, include_index=True):
        k = self.obtain_kmer_feature_for_one_sequence(True)
        k = [int(elem) for elem in k]
        k = [str(elem) for elem in k]
        if include_index:
            k.insert(0, str(self.id))
        return k

class mRNA(kmer_featurization):
    def __init__(self, seq_record):
        super().__init__(k=4)
        self.seq = seq_record.seq.back_transcribe().upper()
        self.id = seq_record.id
        self.max_mrna_len = 9756

    def reduce_mrna(self, rnai, extension=500, rnai_size=None, pad=False):
        if rnai_size is None:
            rnai_match = rnai.seq.reverse_complement().back_transcribe().upper()
        else:
            rnai_match = (
                rnai.seq[:rnai_size].reverse_complement().back_transcribe().upper()
            )
        self.max_mrna_size = 2 * extension + len(rnai_match)
        rna_index = self.seq.find(rnai_match)
        mrna_start = rna_index - extension
        mrna_end = rna_index + len(rnai_match) + extension
        sequence_length = len(self.seq)
        self.seq = self.seq[max(mrna_start, 0) : min(mrna_end, sequence_length)]
        if pad:
            for _ in range(mrna_start, 0):
                self.seq = Seq("N") + self.seq
            for _ in range(sequence_length, mrna_end):
                self.seq += Seq("N")

    def flank_mrnas(self, rnai):
        self.reduce_mrna(rnai, extension=19, rnai_size=19, pad=True)
        return mRNA(SeqRecord(seq=self.seq[:19], id=self.id)), mRNA(
            SeqRecord(seq=self.seq[-19:], id=self.id)
        )


class siRNA(kmer_featurization):
    def __init__(self, seq_record_antisense):
        super().__init__(k=3)
        assert (
            len(seq_record_antisense.seq) >= 19
        ), "siRNA is shorter than 19nt which is the minimum"
        self.max_sirna_len = 21
        self.seq = seq_record_antisense.seq.transcribe().upper()[: self.max_sirna_len]
        self.id = seq_record_antisense.id
        self.thermo_row = []
        self.mrna_match = None
        self.mrna_sequence_match = None

    def _rnaup(self, mrna, extension=500, extra=False):
        if extension is not None:
            mrna.reduce_mrna(self, extension=extension)
        couple = str(mrna.seq) + "\n" + str(self.seq)
        # calling RNAup with some specifics
        proc = subprocess.Popen(
            ["RNAup", "-b", "-d2", "--noLP", "-o", "-c", "'S'", "RNAup.out"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        outs, errs = proc.communicate(input=couple.encode(encoding="utf-8"))
        # start = " ("
        # end = ")\n"
        results = outs.decode(encoding="utf-8")
        pattern = r'(?P<mrna_location>\b\d+,\d+\b) *: *(?P<sirna_location>\b\d+,\d+\b) *\((?P<gibbs>-?\d+\.\d+\s*=\s*-?\d+\.\d+(?:\s*\+\s*-?\d+\.\d+)*)\)\n(?P<rna>[AUCGT&]+)'
        matches = re.search(pattern, results, re.MULTILINE)
        if matches is None:
            raise ValueError(f"Regex pattern not found in RNAup results\nPattern: {pattern}\nRNAup results: {results}")
        # gibbs_results = results[results.find(start) + len(start) : results.rfind(end)]
        gibbs_results = matches.group('gibbs')
        gibbs_results = gibbs_results.replace("=", "").replace("+", "")
        dGtotal, dGinteraction, dGmrna_opening, dGrnai_opening = list(
            map(float, gibbs_results.split())
        )
        self.thermo_row.extend([dGtotal, dGmrna_opening, dGrnai_opening])
        if extra:
            mrna_location = matches.group("mrna_location").split(",")
            self.mrna_match = (int(mrna_location[0]), int(mrna_location[1]))
            self.mrna_sequence_match = matches.group("rna").split("&")[0]

    def thermo_features(self, mrna: mRNA, include_index=True):
        if len(self.thermo_row) == 0:
            if include_index:
                self.thermo_row = [self.id, mrna.id]
            self._cal_thermo_feature()
            self._rnaup(mrna)
        return self.thermo_row

    def _cal_thermo_feature(
        self, intermolecular_initiation=4.09, symmetry_correction=0.43
    ):
        """Calculate thermodynamic stability metrics."""
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
        sequence = self.seq[:19]
        sum_stability = 0

        if sequence[0] == "A":
            sum_stability += 0.45
        if sequence[18] == "U":
            sum_stability += 0.45

        for i in range(18):
            bimer = sequence[i : i + 2]
            stability_value = bimer_values.get(bimer, 0)
            self.thermo_row.append(stability_value)
            sum_stability += stability_value

        sum_stability += intermolecular_initiation
        sum_stability += symmetry_correction
        self.thermo_row.append(round(sum_stability, 2))
