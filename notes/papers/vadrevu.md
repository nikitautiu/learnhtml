## Semantic partitioning of web pages - Vadrevu et al. 2005

### Introduction
Does not make the assumption that an ontology of conecept is known apriori abount the data. Instead it defines the semantic roles of *concept*, *attribute* and *value* to generalize the concepts. Does away with the notion that they are template driven.

### Related work
Presents different classes ofsegemntation algorithms. There are those that assume templating of classes, which that author states that is not necessary for their algorithm as it handles noise.

It refers to itself as a *page sgmentation algorithm*, but unlike VIPS it uses **information theoretic methods** to define homogneity of a segment, not heuristics.


### Page segmentation
Does **flat segmentation** as it assumees content resides in leaf ndodes. A segment is defined as a continuous set of leaf nodes which are then split based on their homogenity. This homogenity is defined based on entropy of a root-to-leaf path.

It then employs a sort of clustering algorithm that minimizes entropy of found groups of nodes. ased on this algorithm it generates a sort of
DSL regexes to extract blocks belonging to the same  cluster.

A *root-to-leaf* path is defined in terms of its sequence of nodes(ie. `/html/head/title`).

### Results
Good performance with the TAP dataset with their heirarchy of Concept-Attribute-Value segemntation on the first two levels of it.

```python

```
