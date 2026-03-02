[[Microinvert-project/mayfly_internal/functions|functions]]
**Arguments:**
`long` : a flattened array of pixel values.
`k` : the number of summary statistics to take.
**Returns:**
`summary` : a `(k)` length np.array of summary statistics.
**Description:**
Flexible list of summary statistics for training and making predictions with a naive bayes model.
- Only called by `image_summary`