import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
import os
import joblib


def get_train_sample(n: int = 5, save: bool = False):
    """
    Get a sample of the training data, optionally save.

    Args:
        n (int): Sample size. Defaults to 5.
        save (bool): Option to save sample locally. Defaults to False.

    Returns:
        Dataset: A dataset of n examples, each with keys 'image' (np.ndarray), 
                 'species' (int), and 'count' (int).
    """

    load_dotenv(find_dotenv())
    data = load_dataset("shenandoah-macroinvertebrates/ept-bioassessment-dataset", token=os.getenv("HF_TOKEN"), split="train")
    data = data.with_format("numpy")

    indices = np.random.choice(len(data), size=n, replace=False).tolist()    
    sample = data.select(indices)

    # Optionally save to disk
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
