# NB Helper Functions 

- [get\_slices](#get_slices)
- [grey\_if\_true](#grey_if_true)
- [image\_summary](#image_summary)
- [NB\_pred\_single](#nb_pred_single)
- [NB\_predict\_on\_images](#nb_predict_on_images)
- [pre\_summary](#pre_summary)
- [to\_chunks](#to_chunks)
- [zoom\_out](#zoom_out)

---

## get_slices

**Arguments:**
- `rgb` : the original full image
- `size` : the size of sub-images you want

**Returns:**
- `vertical_slices` : list of tuple pairs of slice objects that index the location of the subimage
- `horizontal_slices`

**Description:**
Helps the [to_chunks](#to_chunks) function.

---

## grey_if_true

**Arguments:**
- `image` : an `np.array` of shape `(H,W,3)`
- `grey` : boolean; if `True` then image becomes grey

**Returns:**
- `image` : the same `np.array` of shape `(H,W,3)`

**Description:**
Converts a typical image to greyscale by averaging the 3 color channels for each pixel.

---

## image_summary

**Arguments:**
- `image` : an image `np.array` of shape `(H,W,3)`
- `k=3` : the number of summary statistics to take

**Returns:**
- `summary` : an `np.array` of shape `(3*k)`

**Description:**
Takes summary statistics of an image to use as features in a NB model.

---

## NB_pred_single

**Arguments:**
- `image` : an image of type `np.array`
- `model` : a trained NB model
- `slice_factor` : positive integer

**Returns:**
- `subimages` : an `np.array` of sub-images, shape `(x*y, 224, 224, 3)`
- `predictions` : an `np.array` of predictions from the NB model, shape `(x*y)`
- `shape` : the tuple `(x, y)`

**Description:**
Breaks a full image into chunks, makes empty/not-empty predictions on each chunk, and returns the chunks and predictions.

---

## NB_predict_on_images

**Arguments:**
- `model` : the Naive Bayes model to make predictions on images
- `images` : an `np.array` of shape `(k, H, W, 3)`
- `M=3` : the number of summary statistics to take — must match what the NB model was trained on

**Returns:**
- `predictions` : an `np.array` of predictions of shape `(k)`, aligned with the indexes of the input images

**Description:**
Takes in a list of images, computes their summary statistics, and makes predictions on them. Called by [NB_pred_single](#nb_pred_single).

---

## pre_summary

**Arguments:**
- `long` : a flattened array of pixel values
- `k` : the number of summary statistics to take

**Returns:**
- `summary` : a `(k)` length `np.array` of summary statistics

**Description:**
Flexible list of summary statistics for training and making predictions with a Naive Bayes model. Only called by [image_summary](#image_summary).

---

## to_chunks

**Arguments:**
- `image` : a full image to convert to chunks; must be `np.array` of shape `(H,W,3)`
- `slice_factor` : a crude method for downscaling the image; positive integer

**Returns:**
- `subimages` : an `np.array` of shape `(x*y, 224, 224, 3)`
- `shape` : the tuple `(x, y)`

**Description:**
Converts an image into `x*y` subimages.

---

## zoom_out

**Arguments:**
- `full_image` : the original full image of shape `(H,W,3)`
- `horizontal_slice` : the horizontal location of the sub-image
- `vertical_slice` : the vertical location of the sub-image
- `size` : the size of the output image

**Returns:**
- `scaled_horizontal_slice` : location of the output image on the full image
- `scaled_vertical_slice`

**Description:**
If a chunk is labeled as non-empty, zooms out so that there is more predictive context for the classifying CNN.
