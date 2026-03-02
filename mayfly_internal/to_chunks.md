[[Microinvert-project/mayfly_internal/functions|functions]]
**Arguments:**
`image` : a full image to convert to chunks. Must be np.array of shape `(H,W,3)`.
`slice_factor` : a crude method for downscaling the image. Positive integer.
**Returns:**
`subimages` : an np.array of shape `(x*y,224,224,3)`.
`shape` the tuple `(x,y)`
**Description:**
Converts an image into `x*y` subimages.