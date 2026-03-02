_visualize_predictions_\
\
**Arguments:**\
`subimages` : an np.array of shape `(x*y, 224, 224, 3)`, list of sub-images.\
`prediction` : an np.array of shape `(x*y)`, list of predictions.\
`shape` : the tuple `(x,y)`.\
**Returns:**\
Nothing.\
**Description:**\
If you want to see the predictions made by a model for a single full image, you can plug in the image and its predictions. This allows us to determine the quality of predictions or any patterns obvious to the human eye.
