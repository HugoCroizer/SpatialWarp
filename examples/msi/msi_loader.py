"""Load vendor MSI (mass spectrometry imaging) export files.

This is deliberately MSI-specific — CSV formats, m/z-to-metabolite mapping,
and other conventions here are particular to this vendor's export format.
See ``../l12_walkthrough.ipynb`` for how to turn the resulting points/features
into a generic ``spatialdata.SpatialData`` object, using the core package's
technology-agnostic primitives (``pick_landmarks``, ``register_elastic``,
``build_spatialdata``).
"""

import pandas as pd


class MSIdata:
    def __init__(self, counts, metabolites, coordinates, sep=";"):
        self.counts = pd.read_csv(counts, comment="#", sep=sep)
        # Not parsed with comment="#" like the other two files: metabolite color codes
        # (e.g. "#00e2e5") contain '#' characters that pandas would otherwise misread as
        # inline comments, truncating those fields. The number of leading SCiLS export
        # comment lines also varies between exports (some have none), so it's counted
        # directly instead of a fixed skiprows.
        self.metabolites = pd.read_csv(metabolites, sep=sep, skiprows=self._count_leading_comment_lines(metabolites))
        self.coordinates = pd.read_csv(coordinates, comment="#", sep=sep)
        self.counts = self.map_metabolites_to_counts()

    @staticmethod
    def _count_leading_comment_lines(path):
        n = 0
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                n += 1
        return n

    def update_coordinates(self, new_x, new_y):
        self.coordinates["x"] = new_x
        self.coordinates["y"] = new_y

    def map_metabolites_to_counts(self):
        """Map each m/z value in ``counts`` to a metabolite name, based on the
        m/z +/- interval ranges in ``metabolites``."""
        # column is named "Interval Width (+/- Da)" in some exports, "IntervalWidth" in others
        interval_col = next(c for c in self.metabolites.columns if "interval" in c.lower())
        count_mz_values = self.counts["m/z"].values

        mz_to_metabolite = {}
        for count_mz in count_mz_values:
            match = None
            for _, met_row in self.metabolites.iterrows():
                if abs(count_mz - met_row["m/z"]) <= met_row[interval_col]:
                    match = met_row["Name"]
                    break
            mz_to_metabolite[count_mz] = match if match is not None else "Unknown"

        result = self.counts.copy()
        result["m/z"] = result["m/z"].map(mz_to_metabolite)
        return result

    def get_count_formated(self):
        """Return a (spots x metabolites) DataFrame with 'x'/'y' columns appended."""
        metabolites = self.counts.iloc[:, 0].values
        spot_ids = self.counts.columns[1:]

        intensity_matrix = self.counts.iloc[:, 1:].T
        intensity_matrix.columns = metabolites
        intensity_matrix.index = spot_ids

        intensity_matrix["x"] = self.coordinates["x"].values
        intensity_matrix["y"] = self.coordinates["y"].values
        return intensity_matrix
