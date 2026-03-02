[[Microinvert-project/mayfly_internal/functions|functions]]
**Arguments:**
`image` : an np.array of shape `(H,W,3)`.
`grey` : boolean, if True then image becomes grey
**Returns:**
`image` : the same np.array of shape `(H,W,3)`
**Description:**
Just converts a typical image to greyscale by averaging the 3 channels of color for each pixel.