"""Minimal 4-vector helpers (E, px, py, pz), GeV units."""

from __future__ import annotations

import numpy as np


def boost(p4, beta_vec):
    """Boost 4-vector(s) ``p4`` (..., 4) by velocity ``beta_vec`` (3,) or (...,3).

    ``beta_vec`` is the velocity of the *new* frame origin as seen from the old
    frame, i.e. this maps rest-frame quantities into the lab when ``beta_vec``
    is the parent's lab velocity.
    """
    p4 = np.atleast_2d(np.asarray(p4, float))
    b = np.atleast_2d(np.asarray(beta_vec, float))
    b2 = np.sum(b * b, axis=-1)
    b2 = np.clip(b2, 0.0, 1.0 - 1e-16)
    gam = 1.0 / np.sqrt(1.0 - b2)
    e, p = p4[..., 0], p4[..., 1:]
    bp = np.sum(b * p, axis=-1)
    gam2 = np.where(b2 > 0, (gam - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    e_new = gam * (e + bp)
    p_new = p + (gam2 * bp)[..., None] * b + (gam * e)[..., None] * b
    out = np.concatenate([e_new[..., None], p_new], axis=-1)
    return out


def beta_of(p4):
    p4 = np.asarray(p4, float)
    return p4[..., 1:] / p4[..., 0:1]


def mass_of(p4):
    p4 = np.asarray(p4, float)
    m2 = p4[..., 0] ** 2 - np.sum(p4[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.maximum(m2, 0.0))


def make_p4(m, e_tot, direction):
    """4-vector from mass, total energy and a unit direction."""
    p = np.sqrt(max(e_tot * e_tot - m * m, 0.0))
    d = np.asarray(direction, float)
    return np.concatenate([[e_tot], p * d])


def random_unit(rng, n=1):
    c = 2.0 * rng.random(n) - 1.0
    s = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    ph = 2.0 * np.pi * rng.random(n)
    return np.stack([s * np.cos(ph), s * np.sin(ph), c], axis=-1)


def rotate_to(z_axis, v):
    """Rotate vectors ``v`` (..., 3), expressed in a frame whose z is (0,0,1),
    into the frame whose z is the unit vector ``z_axis``."""
    z = np.asarray(z_axis, float)
    z = z / np.linalg.norm(z)
    a = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    v = np.asarray(v, float)
    return v[..., 0:1] * x + v[..., 1:2] * y + v[..., 2:3] * z
