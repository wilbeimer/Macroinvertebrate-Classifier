[[Microinvert-project/mayfly_internal/functions|functions]]
**Arguments:**
`rgb` : the original full image
`size` : the size of sub-images you want
**Returns:**
`vertical_slices` : List of tuple pairs of slice objects that index the location of the subimage.
`horizontal_slices` : 
**Description:**
Helps the `to_chunk` function.