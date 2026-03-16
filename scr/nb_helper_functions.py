import numpy as np
import itertools


def NB_pred_single(image: np.ndarray, model, slice_factor: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Chunk a single image and return NB predictions and grid shape."""
    subimages, (v, h) = to_chunks(image, slice_factor)
    predictions = NB_predict_on_images(model, subimages)
    return (subimages, predictions, (v-1, h-1))


def to_chunks(image: np.ndarray, slice_factor: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Downsample an image by slice_factor and split into 224x224 chunks.

    Raises:
        TypeError: If image is not 3-dimensional.
    """
    sliced_image = image[::slice_factor, ::slice_factor]
    if (len(image.shape) != 3):
        raise TypeError(f"Wrong shape of array!!! You have {image.shape}")
    vertical_slices, horizontal_slices = get_slices(sliced_image, 224)
    product = np.array(list(itertools.product(vertical_slices[:-1], horizontal_slices[:-1])))
    slices = np.empty(len(product), dtype=object)
    slices[:] = list(map(tuple, product))
    subimages = np.array([sliced_image[chunk] for chunk in slices])
    return subimages, (len(vertical_slices), len(horizontal_slices))


def to_chunks(image: np.ndarray, slice_factor: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Downsample an image by slice_factor and split into 224x224 chunks.

    Raises:
        TypeError: If image is not 3-dimensional.
    """
    # Downsample image by taking every slice_factor-th pixel
    sliced_image = image[::slice_factor, ::slice_factor]
    
    if (len(image.shape) != 3):
        raise TypeError(f"Wrong shape of array!!! You have {image.shape}")
        
    vertical_slices, horizontal_slices = get_slices(sliced_image, 224)
    
    # Get all combinations of vertical and horizontal slices
    product = np.array(list(itertools.product(vertical_slices[:-1], horizontal_slices[:-1])))
    
    # Convert to tuple pairs so they can be used as 2D array indices
    slices = np.empty(len(product), dtype=object)
    slices[:] = list(map(tuple, product))
    
    subimages = np.array([sliced_image[chunk] for chunk in slices])
    return subimages, (len(vertical_slices), len(horizontal_slices))


def pre_summary(long: np.ndarray, k: int) -> np.ndarray:
    """
    Compute a statistical summary vector for a 1D array of pixel values.

    Computes k central moments, k raw moments, and 3 extreme value statistics,
    all normalized by standard deviation. Used as features for the NB model.

    Args:
        long (np.ndarray): 1D array of pixel values for a single channel.
        k (int): Number of moments to compute.

    Returns:
        np.ndarray: Summary vector of length 2k + 3.
    """
    # Compute central moments (2nd through kth), normalized by std
    central_moments = np.array(list(map(lambda n: np.mean((long - np.mean(long))**n)/(np.std(long)**n), range(2, k+1))))
    
    # Compute raw moments (1st through kth), normalized by std
    raw_moments = np.array(list(map(lambda n: np.mean(long**n)/(np.std(long)**n), range(1, k+1))))
    
    # Compute extreme value statistics: range, distance from max to mean, distance from min to mean
    extreme = np.array([max(long)-min(long), max(long)-raw_moments[0], min(long)-raw_moments[0]])

    summary = np.concatenate((central_moments, raw_moments, extreme), axis=0)
    return summary


def image_summary(image: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute a statistical summary vector for an RGB image chunk.

    Separates the image into R, G, B channels and computes pre_summary
    for each, then concatenates into a single feature vector.

    Args:
        image (np.ndarray): RGB image chunk, shape (H, W, 3).
        k (int): Number of moments to compute per channel. Defaults to 3.

    Returns:
        np.ndarray: Feature vector of length 3 * (2k + 3).
    """
    pixels = image.flatten().tolist()  # flattens the image into a long list

    # separate each color into its own list
    reds = np.array(pixels[0::3])
    greens = np.array(pixels[1::3])
    blues = np.array(pixels[2::3])
    
    red_summary = pre_summary(reds, k)
    blue_summary = pre_summary(blues, k)
    green_summary = pre_summary(greens, k)
    summary = np.concatenate((red_summary, blue_summary, green_summary), axis=0)
    return summary


def NB_predict_on_images(model, images: np.ndarray, M: int = 3) -> np.ndarray:
    """Run NB predictions directly on an array of image chunks."""
    # Compute summary features for each chunk
    summarylist = np.array([image_summary(x, k=M) for x in images])
    predictions = model.predict(summarylist)
    return predictions


def get_slices(rgb: np.ndarray, size: int) -> tuple[list, list]:
    """Generate vertical and horizontal slice indices for chunking an image into size x size grids."""
    height = rgb.shape[0]
    width = rgb.shape[1]
    vertical_slices = []
    horizontal_slices = []
    K = height // size
    for k in range(K):
        t = slice(k * size, min([(k + 1) * size, height - 1]))
        vertical_slices.append(t)
    N = width // size
    for n in range(N):
        t = slice(n * size, min([(n + 1) * size, width - 1]))
        horizontal_slices.append(t)
    return (vertical_slices, horizontal_slices)
    

def zoom_out(FullImage: np.ndarray, horizontal_slice: slice, vertical_slice: slice, sizex: int) -> tuple[slice, slice]:
    """
    Expand a 224x224 chunk slice to a larger sizex x sizex region, clamped to image boundaries.

    Grows the slice equally in all directions by (sizex - 224) // 2 pixels,
    then shifts it if it goes out of bounds. Assumes image shape (6744, 4502, 3).

    Args:
        FullImage (np.ndarray): The full source image (unused, reserved for future use).
        horizontal_slice (slice): Original horizontal slice of the 224x224 chunk.
        vertical_slice (slice): Original vertical slice of the 224x224 chunk.
        sizex (int): Target size to zoom out to.

    Returns:
        tuple[slice, slice]: Expanded (horizontal_slice, vertical_slice) clamped to image bounds.
    """
    changer = (sizex - 224)//2  # This is the amount we'll subtract from the slice starts and add to slice ends.
    original_x = horizontal_slice
    original_y = vertical_slice

    NewYStart = int(original_y.start - changer)
    NewYEnd = int(original_y.stop + changer)
    NewXStart = int(original_x.start - changer)
    NewXEnd = int(original_x.stop + changer)

    # These if statements are used to shift the slices if they go off the edges. PLEASE NOTE THAT OUR IMAGES HAVE SHAPE (6744, 4502, 3)

    if NewYStart < 0:
        NewYEnd -= NewYStart
        NewYStart = 0

    if NewYEnd > 6744:
        NewYStart -= (NewYEnd - 6744)
        NewYEnd = 6744

    if NewXStart < 0:
        NewXEnd -= NewXStart
        NewXStart = 0

    if NewXEnd > 4502:
        NewXStart -= (NewXEnd - 4502)
        NewXEnd = 4502

    ScaledXSlice = __builtins__.slice(NewXStart, NewXEnd)
    ScaledYSlice = __builtins__.slice(NewYStart, NewYEnd)

    return (ScaledXSlice, ScaledYSlice)


def grey_if_true(image: np.ndarray, grey: bool) -> np.ndarray:
    """Return the image greyscaled if grey is True, otherwise return it unchanged."""
    if not grey:
        return image
    elif grey:
        grey_single = image.mean(axis=2, keepdims=True)
        grey_3channel = np.repeat(grey_single, 3, axis=2).astype(image.dtype)
        return grey_3channel
