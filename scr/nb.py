import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scr.nb_helper_functions as nb
from IPython.display import clear_output


def hand_labeling(NpArray: np.ndarray) -> list[int]:
    """
    Interactively label an array of images as empty or non-empty.

    Displays each image one at a time and prompts the user to label it.
    Supports navigating back to previous images and early exit.

    Args:
        NpArray (np.ndarray): Array of images to label, shape (N, H, W, 3).

    Returns:
        list[int]: Labels for each reviewed image. 1 = empty, 0 = non-empty.
                   May be shorter than NpArray if user exits early.

    Controls:
        y - label as empty
        n - label as non-empty
        b - go back to previous image
        c - exit early
    """
    
    labels = [None] * len(NpArray)
    i = 0

    while i < len(NpArray):
        plt.axis("off")
        plt.imshow(NpArray[i])
        plt.show()

        user_input = input(f"[{i+1}] Is this empty? (y= yes, n= no, b=back, c=exit): ").lower()

        plt.clf()
        if (user_input) == 'y' or user_input == 'n':
            labels[i] = 1 if user_input == 'y' else 0
            i += 1
        elif (user_input) == 'b':
            if i > 0:
                i-=1
            else:
                print("You are already at the first image")
        elif user_input == 'c':
            break
        else:
            print("Please give a proper input: (y= yes, n= no, b=back, c=exit): ")
    return labels[:i]

def write_non_empty():
    pass


def nb_assisted_hand_labeling():
    pass


def image_summary():
    pass


def NB_predictions(images: np.ndarray, model, slice_factor: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Run Naive Bayes predictions on a set of images by chunking each one.

    Breaks each image into subimages using the given slice factor, runs the
    model on each chunk, and stacks the results.

    Args:
        images (np.ndarray): Array of images, shape (N, H, W, 3).
        model: Trained scikit-learn Naive Bayes model.
        slice_factor (int): Factor by which to downsample images before chunking.

    Returns:
        tuple:
            subimages (np.ndarray): All chunks, shape (N, n_chunks, 224, 224, 3).
            predictions (np.ndarray): Model predictions per chunk, shape (N, n_chunks).
            shape (tuple): Grid dimensions (n_vertical, n_horizontal).
    """
    subimage_list = []
    prediction_list = []

    for image in tqdm(images, desc="convoluting transpositions"):
        subimages, predictions, shape = nb.NB_pred_single(image, model, slice_factor)
        subimage_list.append(subimages)
        prediction_list.append(predictions)
    subimages = np.stack(subimage_list, axis=0)
    predictions = np.stack(prediction_list, axis=0)
    return subimages, predictions, shape


def adjust_nb_predictions(images: np.ndarray, predictions: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Interactively review and correct model predictions on a set of images.

    Filters chunks predicted as non-empty (1) and presents them for human
    review. After labeling, displays the full prediction grid for each image
    and asks the user if any need relabeling. Recursively relabels low quality
    images until the user is satisfied.

    Args:
        images (np.ndarray): Array of chunked images, shape (N, n_chunks, 224, 224, 3).
        predictions (np.ndarray): Model predictions per chunk, shape (N, n_chunks).
        shape (tuple[int, int]): Grid dimensions (n_vertical, n_horizontal).

    Returns:
        np.ndarray: Updated predictions after human review, shape (N, n_chunks).

    Controls:
        y - label as empty
        n - label as non-empty
        b - go back to previous image
        c - exit early
    """

    photo_indexed_review_list = [images[i][np.where(predictions[i] == 1)] for i in range(images.shape[0])]
    review_list = np.concatenate(photo_indexed_review_list, axis=0)
    original_indexes = []
    for k in range(images.shape[0]):
        for i in np.where(predictions[k] == 1)[0]:
            original_indexes.append((k, i))

    updated_labels = []
    i = 0
    while i < len(review_list):

        # show the image
        clear_output(wait=True)
        plt.clf()
        plt.axis("off")
        plt.imshow(review_list[i])
        plt.show()

        # ask user to label
        print(f"Progress: ({len(updated_labels)}/{review_list.shape[0]})")
        user_inp = input("Is this empty(y/n)?")

        if user_inp == "c":
            break
        elif user_inp in ["y", "n"]:
            if user_inp == "y":
                updated_labels.append(0)
            elif user_inp == "n":
                updated_labels.append(1)
            i += 1

        elif user_inp == "b":
            i -= 1
            updated_labels.pop()
            continue
            clear_output(wait=True)
            plt.clf()
            plt.axis("off")
            plt.imshow(review_list[i])
            plt.show()

    # updates all with the humans labels
    updated_labels = np.array(updated_labels)
    for i in range(updated_labels.shape[0]):
        predictions[original_indexes[i]] = updated_labels[i]

    # the viewer can review their labels
    label_quality = []
    for j in range(predictions.shape[0]):
        # show the image

        view = visualize_predictions(images[j], predictions[j], shape)

        clear_output(wait=True)
        plt.clf()
        plt.axis("off")
        plt.imshow(view)
        plt.show()

        while True:
            quality = input(f"Should this be relabelled?(y/n)   {j}/{predictions.shape[0]}")
            if quality == "n":
                label_quality.append(0)  # 0 means /abeling is not necessary
                break
            elif quality == "y":
                label_quality.append(1)  # 1 means relabeling is necessary
                break
    label_quality = np.array(label_quality)

    low_quality_indexes = np.where(label_quality == 1)
    if low_quality_indexes[0].shape[0] > 0:
        relabeled = label_imageset(images[low_quality_indexes], predictions[low_quality_indexes], shape)
        predictions[low_quality_indexes] = relabeled
    return predictions


def visualize_predictions(subimages: np.ndarray, predictions: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Visualize model predictions by greying out empty chunks on the full image.

    Reshapes chunks back into a grid, greys out chunks predicted as empty,
    and stitches them back into a single full image.

    Args:
        subimages (np.ndarray): Array of image chunks, shape (n_chunks, 224, 224, 3).
        predictions (np.ndarray): Model predictions per chunk, shape (n_chunks,).
        shape (tuple[int, int]): Grid dimensions (n_vertical, n_horizontal).

    Returns:
        np.ndarray: Full reconstructed image with empty chunks greyed out,
                    shape (n_vertical * 224, n_horizontal * 224, 3).
    """
    
    subimages = subimages.reshape(shape[0], shape[1], 224, 224, 3)
    predictions = predictions.reshape(shape[0], shape[1])
    k = 0
    while k < (shape[0]):
        n = 0
        while n < (shape[1]):
            subimages[k, n] = nb.grey_if_true(subimages[k, n], not bool(predictions[k, n]))
            n += 1
        k += 1
    x, y, h, w, c = subimages.shape
    full = subimages.transpose(0, 2, 1, 3, 4).reshape(x * h, y * w, c)
    return full
