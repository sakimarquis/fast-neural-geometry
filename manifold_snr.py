"""This module implements geometric analysis methods from: Sorscher et al. (2022)

Key geometric properties computed:
- Center: manifold centroid
- Radius spectrum (R): SVD singular values capturing manifold spread
- Dimensionality (D): participation ratio = (ΣR²)² / ΣR⁴
- Normalized distance (dist_norm): inter-class distance scaled by radius
- Center-subspace overlap (csa): alignment of manifold subspace with inter-center direction
- Subspace-subspace overlap (ss): overlap between class subspaces
- Bias: asymmetry between manifold radii
- Signal-to-Noise Ratio (SNR): predicts classification separability
"""

import numpy as np
from scipy.special import erfc


def snr_to_error(snr: np.ndarray) -> np.ndarray:
    """Convert SNR to expected error rate [0, 0.5] (0.5 = chance level) via Gaussian CDF."""
    return 0.5 * erfc(snr / np.sqrt(2))


def extract_manifold_geometry(manifold: np.ndarray) -> dict:
    """Extract geometric properties from a single class manifold via SVD. Characterize the "shape" of
    a class's representation cloud. SVD of centered data gives principal directions and their magnitudes.

    :param manifold: (n_samples, n_features) - samples from one class
    """
    center = manifold.mean(axis=0)
    manifold = manifold - center
    _, radii, subspace = np.linalg.svd(manifold, full_matrices=False)

    # Participation ratio: measures "effective" number of dimensions used
    # D=1 means all variance in one direction, D=n means uniform spread
    radii_squared = radii ** 2
    sum_r2 = radii_squared.sum()
    sum_r4 = (radii_squared ** 2).sum()
    dimensionality = sum_r2 ** 2 / sum_r4 if sum_r4 > 0 else 1.0
    return {
        'center': center,  # manifold centroid (the class prototype)
        'radii': radii,  # singular values (variance along each principal direction)
        'subspace': subspace,  # principal directions (right singular vectors V)
        'dimensionality': dimensionality,  # effective dimension D = (ΣR²)² / ΣR⁴ (participation ratio)
        'total_variance': sum_r2  # total spread of the manifold
    }


def compute_center_subspace_overlap(radii: np.ndarray, subspace: np.ndarray, inter_class_direction: np.ndarray) -> float:
    """Measure how much the manifold's spread aligns with the classification axis.

    Project inter-class direction onto each principal component, weight by variance.
    High overlap = manifold extends along the decision boundary = more noise.
    Formula: csa = Σ(cos²θ · R²) / ΣR² where θ is angle to each principal direction.

    :param radii: singular values (spread in each principal direction)
    :param subspace: principal directions (n_components, n_features)
    :param inter_class_direction: unit vector pointing between class centers
    :return: Overlap in [0, 1]. Higher = worse for classification.
    """
    # How aligned is inter-class direction with each principal component?
    cos_theta = subspace @ inter_class_direction  # projection onto each PC
    cos_theta_squared = cos_theta ** 2

    radii_squared = radii ** 2
    total_variance = radii_squared.sum()

    # Variance-weighted average of squared cosines
    if total_variance > 0:
        overlap = (cos_theta_squared * radii_squared).sum() / total_variance
    else:
        overlap = 0.0
    return overlap


def compute_subspace_overlap(radii_a: np.ndarray, subspace_a: np.ndarray, radii_b: np.ndarray, subspace_b: np.ndarray) -> float:
    """Measure how much the two classes' principal subspaces overlap (similarity of noise structure).

    Pairwise cosines between all principal directions, weight by variances.
    High overlap = similar noise directions = harder to separate classes.
    Formula: ss = Σ(cos²φ · Ra² · Rb²) / (ΣRa²)²
    """
    # Cosine of angles between all pairs of principal directions
    cos_phi = subspace_a @ subspace_b.T  # (n_comp_a, n_comp_b)
    radii_a_squared = radii_a ** 2
    radii_b_squared = radii_b ** 2

    # Weight by product of variances in each direction pair
    weighted_overlap = np.einsum('ij,i,j->', cos_phi ** 2, radii_a_squared, radii_b_squared)
    normalization = radii_a_squared.sum() ** 2

    return weighted_overlap / normalization if normalization > 0 else 0.0


def pairwise_manifold_snr(manifold_0: np.ndarray, manifold_1: np.ndarray, m: int = 1) -> dict:
    """Calculates the Signal-to-Noise Ratio (SNR) and geometric properties for binary classification.

    The SNR is derived from the balance of class separation (signal) and manifold overlap (noise):
        signal = dist_norm² + bias/m
        noise² = 1/(D·m) + css + ss/m
        SNR    = 0.5 * signal / noise

    Terms:
        - dist_norm : Normalized inter-class distance (primary signal strength).
        - bias      : Asymmetry in class variances; scales with 1/m.
        - D         : Effective dimensionality (noise reduction via averaging).
        - css       : Center-subspace overlap (manifold spread along the decision axis).
        - ss        : Subspace-subspace overlap (shared noise structure).
        - m         : Training examples per class (m=1 indicates one-shot learning).
    """
    geom_0 = extract_manifold_geometry(manifold_0)
    geom_1 = extract_manifold_geometry(manifold_1)

    center_diff = geom_0['center'] - geom_1['center']
    raw_distance = np.linalg.norm(center_diff)
    inter_class_direction = center_diff / raw_distance if raw_distance > 0 else center_diff

    # 1. Center-subspace overlap: how much does each manifold extend along decision axis?
    csa_0 = compute_center_subspace_overlap(geom_0['radii'], geom_0['subspace'], inter_class_direction)
    csa_1 = compute_center_subspace_overlap(geom_1['radii'], geom_1['subspace'], inter_class_direction)
    # 2. Subspace-subspace overlap: do classes share the same noise directions?
    ss_overlap = compute_subspace_overlap(geom_0['radii'], geom_0['subspace'], geom_1['radii'], geom_1['subspace'])
    # 3. Bias: asymmetry in manifold sizes (larger class dominates prototype)
    variance_ratio = geom_0['total_variance'] / geom_1['total_variance'] if geom_1['total_variance'] > 0 else 0.0
    bias = variance_ratio - 1
    # 4. Dimensionality: higher D = more directions to average over = less noise
    mean_dimensionality = (geom_0['dimensionality'] + geom_1['dimensionality']) / 2
    avg_variance_0 = geom_0['total_variance'] / manifold_0.shape[0]
    normalized_distance = raw_distance / np.sqrt(avg_variance_0) if avg_variance_0 > 0 else 0.0
    css_term = (csa_0 + csa_1 / m) * normalized_distance ** 2
    signal = normalized_distance ** 2 + bias / m
    noise_variance = 1 / (mean_dimensionality * m) + css_term + ss_overlap / m
    noise = np.sqrt(noise_variance) if noise_variance > 0 else 1e-10
    snr = 0.5 * signal / noise
    predicted_error = snr_to_error(snr)

    return {'dimensionality_0': geom_0['dimensionality'], 'dimensionality_1': geom_1['dimensionality'],
            'dimensionality_mean': mean_dimensionality, 'dist_norm': normalized_distance,
            'center_subspace_overlap_0': csa_0, 'center_subspace_overlap_1': csa_1, 'subspace_overlap': ss_overlap,
            'bias': bias, 'signal': signal, 'noise': noise, 'snr': snr, 'predicted_error': predicted_error}
