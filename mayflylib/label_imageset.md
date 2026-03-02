[[functions]]
**Arguments:**
`images` : an np.array of shape `(n,x*y,224,224,3)` of sub-images.
`predictions` : an np.array of shape `(n,x*y)` of the NB predicted values.
`shape` : the tuple `(x,y)`.
**Returns:**
`labels` : an np.array of shape `(n,x*y)` of the human adjusted NB predictions.
**Description**
The NB model is assumed to have good recall, but poor precision. In order to train a new model to improve the precision, the positive predictions of the NB model must be labeled. 
