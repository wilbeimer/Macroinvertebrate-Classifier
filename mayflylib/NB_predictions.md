_NB_predictions_\
\
**Arguments:**\
`images` : An np.array of shape `(n,H,W,3)`. Essentially a list of RGB images.\
`model` : A Naive Bayes Model that uses the summary statistics of a sub-image to make predictions on empty cells.\
`slice_factor` : A crude method of downscaling the image, set to 1 to keep image original size, increase to downscale.\
**Returns:**\
`subimages` : An np.array of shape `(n,x*y,224,224,3)` matrix of sub-images.\
`predictions` : An np.array of shape `(n,x*y)` matrix of prediction for all subimages.\
`shape` : the tuple `(x,y)`.\
**Description:**\
`NB_predictions` is used to convert a np.array of full images into a np.array of sub-images with corresponding labels where the model predicted empty space. 
