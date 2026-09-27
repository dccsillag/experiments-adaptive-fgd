import os

# Create unverified context to bypass certificate check
import ssl

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.datasets import load_svmlight_file
from ucimlrepo import fetch_ucirepo

ssl._create_default_https_context = ssl._create_unverified_context


def get_sepsis_data():
    if os.path.exists("data/sepsis_X.npy") and os.path.exists("data/sepsis_Y.npy"):
        x_full = np.load("data/sepsis_X.npy")
        y_full = np.load("data/sepsis_Y.npy")
    else:
        dataset = fetch_ucirepo(id=827)
        x_full = dataset.data.features.to_numpy()
        y_full = dataset.data.targets["hospital_outcome_1alive_0dead"].to_numpy()

        np.save("data/sepsis_X.npy", x_full, allow_pickle=False)
        np.save("data/sepsis_Y.npy", y_full, allow_pickle=False)

    return x_full, y_full


def get_gas_turbine_data():
    if os.path.exists("data/gas_X.npy") and os.path.exists("data/gas_Y.npy"):
        x_full = np.load("data/gas_X.npy")
        y_full = np.load("data/gas_Y.npy")
    else:
        dataset = fetch_ucirepo(id=551)
        x_full = dataset.data.features.drop(["TEY"], axis=1).to_numpy()
        y_full = dataset.data.features["TEY"].to_numpy()

        np.save("data/gas_X.npy", x_full, allow_pickle=False)
        np.save("data/gas_Y.npy", y_full, allow_pickle=False)

    return x_full, y_full


def get_nutrition_data():
    if os.path.exists("data/nutrition_X.npy") and os.path.exists(
        "data/nutrition_Y.npy"
    ):
        x_full = np.load("data/nutrition_X.npy")
        y_full = np.load("data/nutrition_Y.npy")
    else:
        dataset = fetch_ucirepo(id=887)
        x_full = dataset.data.features.to_numpy()
        y_full = (dataset.data.targets["age_group"] == "Adult").to_numpy()

        np.save("data/nutrition_X.npy", x_full, allow_pickle=False)
        np.save("data/nutrition_Y.npy", y_full, allow_pickle=False)

    return x_full, y_full


def get_synthetic_data():
    N = 1_000

    samples = jax.random.uniform(jax.random.key(0), (N, 2))

    return samples, samples[:, 0] * jnp.sin(20 * samples[:, 0])


def get_codrna_data():
    x, y = load_svmlight_file("data/cod-rna.dat")
    x = x.todense()

    # TODO standardize x?
    y = (y + 1) / 2

    return x, y
