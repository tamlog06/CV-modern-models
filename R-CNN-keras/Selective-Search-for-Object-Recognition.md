# Summery of Selective Search for Object Recognition
'Selective Search for Object Recognition' is referred in R-CNN for Region proposal.
In order to implement Region proposal, we need to understand the original paper of 'Selective Search for Object Recognition'.

## What
Selective Search is an algorithm for generating all object locations in the image. \
R-CNN use this algorithm to generate the region proposal.

## How
There are mainly two process of Selective Search.

1. Use hierarchical algorithm, proposed in 'Efficient Graph Based Image Segmentation' to create initial regions.
2. Merge nearly regions by the below two steps until group become the whole image.
    1. Calculate all neighboring region's similarities.
    2. Merge the most similar regions.

Similarities between neighboring regions are calcurated by $S(r_i, r_j)$, $r_i$ and $r_j$ represents regions.
The function $S(r_i, r_j)$ is a linear sum of four different similarity functions.

1. $S_{color}(r_i, r_j) = \sum_{k=1}^{n}min(c_i^k, c_j^k)$, $C_i = \{c_i^1, ..., c_i^n \}$
2. $S_
