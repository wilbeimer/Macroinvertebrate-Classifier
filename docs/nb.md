# NB Functions

- [NB\_predictions](#nb_predictions)
- [label\_imageset](#label_imageset)
- [visualize\_predictions](#visualize_predictions)

---

## NB_predictions

**Arguments:**
- `images` : an `np.array` of shape `(n, H, W, 3)` — a list of RGB images
- `model` : a Naive Bayes model that uses summary statistics of a sub-image to predict empty cells
- `slice_factor` : a crude method of downscaling the image; set to `1` to keep original size, increase to downscale

**Returns:**
- `subimages` : an `np.array` of shape `(n, x*y, 224, 224, 3)` — matrix of sub-images
- `predictions` : an `np.array` of shape `(n, x*y)` — predictions for all sub-images
- `shape` : the tuple `(x, y)`

**Description:**
Converts an `np.array` of full images into an `np.array` of sub-images with corresponding labels where the model predicted empty space.

---

## label_imageset

**Arguments:**
- `images` : an `np.array` of shape `(n, x*y, 224, 224, 3)` of sub-images
- `predictions` : an `np.array` of shape `(n, x*y)` of NB predicted values
- `shape` : the tuple `(x, y)`

**Returns:**
- `labels` : an `np.array` of shape `(n, x*y)` of human-adjusted NB predictions

**Description:**
The NB model is assumed to have good recall but poor precision. In order to train a new model to improve precision, the positive predictions of the NB model must be labeled.

---

## visualize_predictions

**Arguments:**
- `subimages` : an `np.array` of shape `(x*y, 224, 224, 3)` — list of sub-images
- `prediction` : an `np.array` of shape `(x*y)` — list of predictions
- `shape` : the tuple `(x, y)`

**Returns:**
Nothing.

**Description:**
Displays predictions made by a model for a single full image. Useful for assessing prediction quality or spotting patterns obvious to the human eye.
