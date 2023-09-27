"""
Modified: 4/24/2023
Abdul Lateef FNU
"""

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import GenericKernelMixin
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter


class SequenceKernel(GenericKernelMixin, Kernel):
    """
    This kernel calculates a similarity score for each peptide sequence by comparing it with every other peptide sequence.
    Matching & similar amino acids increase score.
    """

    def __init__(self, baseline_similarity=0.5, baseline_similarity_bounds=(1e-32, 1)):
        self.baseline_similarity = baseline_similarity
        self.baseline_similarity_bounds = baseline_similarity_bounds

    @property
    def hyperparameter_baseline_similarity(self):
        return Hyperparameter("baseline_similarity", "numeric", self.baseline_similarity_bounds)

    def _f(self, s1, s2):
        """Calculates similarity score for two sequences s1 and s2"""
        return sum(1.0 if c1 == c2 else self.baseline_similarity for c1, c2 in zip(s1, s2))

    def _g(self, s1, s2):
        """Calculates the gradient of the similarity score for two sequences s1 and s2"""
        return sum(0.0 if c1 == c2 else 1.0 for c1, c2 in zip(s1, s2))

    def __call__(self, X, Y=None, eval_gradient=False):
        """Computes the kernel matrix or kernel value"""
        if Y is None:
            Y = X

        if eval_gradient:
            return np.array([[self._f(x, y) for y in Y] for x in X]), \
                np.array([[[self._g(x, y)] for y in Y] for x in X])
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X)"""
        return np.array([self._f(x, x) for x in X])

    def is_stationary(self):
        """Returns whether the kernel is stationary"""
        return False

    def clone_with_theta(self, theta):
        """Returns a clone of this kernel instance with given theta"""
        cloned = clone(self)
        cloned.theta = theta
        return cloned


# Helper function to calculate amino acid composition
def get_aa_composition(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    composition = [sequence.count(aa) / len(sequence) for aa in amino_acids]
    return composition


# Helper function to calculate molecular weight
def get_molecular_weight(sequence):
    aa_weights = {
        'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18,
        'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.17, 'L': 113.16,
        'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19,
        'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18
    }
    return sum(aa_weights[aa] for aa in sequence)


def extract_features(sequence):
    aa_composition = get_aa_composition(sequence)
    molecular_weight = get_molecular_weight(sequence)
    return np.array([*aa_composition, molecular_weight])
