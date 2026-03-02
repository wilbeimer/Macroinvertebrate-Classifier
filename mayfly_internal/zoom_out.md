_zoom_out_\
\
**Arguments:**\
`full_image` : the original full image shaped `(H,W,3)`\
`horizontal_slice` : the horizontal location of the sub-image\
`vertical_slice` : the vertical location of the sub-image\
`size` : the size of the output image.\
**Returns:**\
`scaled_horizontal_slice` location of the output image on the full image. \
`scaled_vertical_slice`\
**Description:**\
If a chunk is labeled as non-empty, we want the model to "zoom out" so that there is more predictive information for the classifying CNN. 
