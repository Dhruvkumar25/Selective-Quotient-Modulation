"""Core operators used by the Selective-Quotient Modulation experiments.

This file intentionally separates the deterministic signal model from expensive
training/evaluation loops. It is sufficient to reconstruct the exact fractional-
delay nuisance dictionaries, selective projectors, and projection decoder.
"""
from __future__ import annotations
import numpy as np

TDLA_DELAY = np.array([
    0.0000,0.3819,0.4025,0.5868,0.4610,0.5375,0.6708,0.5750,0.7618,
    1.5375,1.8978,2.2242,2.1718,2.4942,2.5119,3.0582,4.0810,4.4579,
    4.5695,4.7966,5.0066,5.3043,9.6586], dtype=float)
TDLA_POWER_DB = np.array([
    -13.4,0,-2.2,-4,-6,-8.2,-9.9,-10.5,-7.5,-15.9,-6.6,-16.7,-12.4,
    -15.2,-10.8,-11.3,-12.7,-16.2,-18.3,-18.9,-16.6,-19.9,-29.7], dtype=float)


def centering_matrix(L: int) -> np.ndarray:
    return np.eye(L) - np.ones((L,L))/L


def fractional_delay_matrix(L: int, delay_samples: float, half: int = 8,
                            beta: float = 8.0) -> np.ndarray:
    n = np.arange(L)[:,None]
    k = np.arange(L)[None,:]
    u = n-k-delay_samples
    s = np.sinc(u)
    w = np.zeros_like(u)
    mask = np.abs(u) <= half
    z = np.sqrt(np.maximum(0.0, 1-(u[mask]/half)**2))
    w[mask] = np.i0(beta*z)/np.i0(beta)
    return s*w


def tdla_operators(L: int, delay_scale_ns: float = 30.0,
                   sample_period_ns: float = 100.0):
    delays = TDLA_DELAY*delay_scale_ns/sample_period_ns
    powers = 10**(TDLA_POWER_DB/10)
    powers = powers/powers.sum()
    D = np.stack([fractional_delay_matrix(L,d) for d in delays])
    return delays,powers,D


def weighted_nuisance_dictionary(x: np.ndarray, omega: float,
                                 powers: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Real power-weighted TDL nuisance dictionary, centered in time."""
    L = x.size
    H = centering_matrix(L)
    phase = np.exp(1j*omega*np.arange(L))
    rows = []
    for p in range(len(powers)):
        v = (D[p]@x)*phase
        a = np.sqrt(powers[p])
        rows.extend([a*v.real, a*v.imag])
    return np.stack(rows)@H


def selective_basis(x: np.ndarray, omega: float, rank: int,
                    powers: np.ndarray, D: np.ndarray) -> np.ndarray:
    F = weighted_nuisance_dictionary(x,omega,powers,D)
    _,_,Vh = np.linalg.svd(F,full_matrices=False)
    return Vh[:rank].T


def selective_projector(x: np.ndarray, omega: float, rank: int,
                        powers: np.ndarray, D: np.ndarray) -> np.ndarray:
    Q = selective_basis(x,omega,rank,powers,D)
    return Q@Q.T


def score_received(Y: np.ndarray, Q: np.ndarray) -> float:
    """Projection energy ||Y_c Q||_F^2 using an orthonormal basis Q."""
    Yc = Y-Y.mean(axis=-1,keepdims=True)
    return float(np.sum((Yc@Q)**2))


def exact_fractional_channel(x_batch: np.ndarray, h: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Apply exact finite-block fractional TDL operator.
    x_batch: (B,L) complex, h: (B,P) complex.
    """
    y = np.zeros_like(x_batch,dtype=complex)
    for p in range(h.shape[1]):
        y += h[:,p,None]*(x_batch@D[p].T)
    return y


def random_tdla_coefficients(rng: np.random.Generator, B: int,
                             powers: np.ndarray) -> np.ndarray:
    z = (rng.normal(size=(B,len(powers))) + 1j*rng.normal(size=(B,len(powers))))/np.sqrt(2)
    return z*np.sqrt(powers)[None,:]


def selective_margin(S: np.ndarray, true_projectors: np.ndarray,
                     wrong_projectors: np.ndarray):
    """Compute epsilon, delta, Delta for a fixed noiseless 2xL received block."""
    energy = np.sum(S*S)
    fit_true = np.einsum('al,klm,am->k',S,true_projectors,S)
    epsilon = np.sqrt(np.maximum(0.0,energy-fit_true)).min()
    fit_wrong = np.einsum('al,klm,am->k',S,wrong_projectors,S)
    delta = np.sqrt(np.maximum(0.0,energy-fit_wrong)).min()
    return float(epsilon), float(delta), float(delta-epsilon)
