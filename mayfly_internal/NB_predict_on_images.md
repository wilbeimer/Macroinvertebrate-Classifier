[[Microinvert-project/mayfly_internal/functions|functions]]
**Arguments:**
`model` : the Naive Bayes model to make predictions on images.
`images` : An np.array of shape `(k,H,W,3)`
`M=3` : The number of summary statistics to take. *Make sure this is the same as the NB model was trained on.*
**Returns:**
`predictions` : an np.array of predictions of shape `(k)`, aligned with the indexes of the input images.
**Description:**
Takes in a list of images, takes their summary statistics, and makes predictions on them. It is called by `NB_pred_single`. 