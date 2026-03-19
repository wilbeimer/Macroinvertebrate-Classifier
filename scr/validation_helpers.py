import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
import os
import joblib


def get_train_sample(n: int = None, save: bool = False, seed: int = None):
    """
    Get a sample of the training data, optionally save.
    Args:
        n (int): Sample size. If None, returns the full dataset. Defaults to None.
        save (bool): Option to save sample locally. Defaults to False.
        seed (int): Random seed for reproducibility. Defaults to None.
    Returns:
        Dataset: A dataset of n examples, each with keys 'image' (np.ndarray), 
                 'species' (int), and 'count' (int).
    """
    load_dotenv(find_dotenv())
    data = load_dataset("shenandoah-macroinvertebrates/ept-bioassessment-dataset", token=os.getenv("HF_TOKEN"), split="train")
    data = data.with_format("numpy")
    if n is None:
        return data
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=n, replace=False).tolist()
    sample = data.select(indices)
    if save:
        np.save("sample_images.npy", np.array(sample, dtype=object))
    return sample

def get_single_image():
    sample = get_train_sample(n=1)
    return sample


def get_nb_model():
    try:
        return joblib.load("models/naive_bayes_model.pkl")
    except FileNotFoundError:
        raise NotImplementedError("Please provide a model at models/naive_bayes_model.pkl")
