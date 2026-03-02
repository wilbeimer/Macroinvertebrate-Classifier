[[Microinvert-project/mayfly_internal/functions|functions]]
**Arguments:**
`image` : an image np.array of shape `(H,W,3)`
`k=3` : the amount of summary statistics to take. 
**Returns:**
`summary` : an np.array of shape `(3*k)` 
**Description:**
Takes summary statistics of an image to use as features in a NB model.