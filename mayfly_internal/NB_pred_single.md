_NB_pred_single_\
\
**Arguments:**\
`image` : an image of type np.array.\
`model` : a train NB model.\
`slice_factor` : Positive integer.\
**Returns:**\
`subimages` : an np.array of sub-images, shape: `(x*y,224,224,3)`\
`predictions` : an np.array of prediction from the NB model, shape : `(x*y)`\
`shape` : the tuple `(x,y)`\
**Description:**\
- Breaks a full image into chunks\
- Makes empty/not empty predictions on each chunk.\
- Returns the chunks and predictions.\
