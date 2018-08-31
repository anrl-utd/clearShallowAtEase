# Resilient Deep Distributed Neural Networks (RDDNNs)
We present multiple methods to retain accuracy in the case of single or multiple layer failure during inference.

Guide to result reproduction:

## Dataset preprocessing
Obtain the multiview camera dataset [here](https://cvlab.epfl.ch/data/multiclass). When unzipped, this data will be split into different folders, with images from each camera (c0, c1, c2, ..., c5), and a folder with bounding boxes for these cameras.

We recommend storing the data in the following hierarchy:
`
.  
+-- multiview  
|   +-- c0  
|   +-- c1  
|   +-- c2  
|   +-- c3  
|   +-- c4  
|   +-- c5  
`  
sss   
