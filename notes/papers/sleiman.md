# A Survey on Region Extractors from Web Documents - Sleiman et al. 2013

Classifies **region extractors** separately from other data extractors. It defines region extractors some sort of preprocessors that are used to preselect data. The more general term, for all of them is *information extractors*.
States the other works exist surveying information extractors. Turmo and Sarawagi found out that they can be split into either heuristical or statistical algorithms. They are not universally applicable.

## Introduction
A *region* is an HTML fragment that shows cohesive information when rendered on a web browser. The purpose of a region extractor, is therefore to actually relieve the information extraction process from having to differentiate between relevant and irrelevant regions.

Therefore:
* Information extractor - extracts structured data
* Region extractor - identifies regions which contain the above information

## Related work
Turmo et al. created a taxonomy for classifying machine learning bsed extractions. They based it on whether it learns rules or statistical models(basically domain specific rules or statistical models). They compared 15 different extraction systems.

## Region extraction
The paper describes a few state-of-the-art proposals for region extractions at that time.

### Embley et  al.
Uses heuristics to determine data regions. Based on finding the highest fan-out DOM node as a the *data region*. Finds then separator nodes via heuristical models, then applies some information theoretical methods to identify the best candidates for separators. It is mainly unsupervised, but to achieve higher performance Butler et al. noted that it need domain-specific ontology to better differentiate data records.

### OMINI
Again, based on the the hypothesis of the unique data region and finds separators, based on heuristics and statistical models. For example, it searches for repeating tags with a small std of its size.

### MDR
Begins with the assumption that a data region contains repetitive information with respect to some HTML tag structure.

### TPC 
Tag path clustering. Based on the hypothesis that a data reguin contains multiple contiguous or noncontiguous data records that are similarily renered, visually aligned and contain at least three HTML tags.

Clusters html paths , using a spectral clustering algorithm on the visual features. The visual features are described using signal theory and their similarity is calculated in this context. Makes some strong assumptions on how many data records should be in the region

### DSE 
Based on comparing documents with similar URLs. Then uses a similarity function to get identical nodes from two documents. Those who are identical, constitute ancilliary regions. Prunes those from the tree.

### U-REST
Describes itself as an unsupervised algorithm. Learns similarities between subtrees of an html document using a SVM on the sample documents. The learned function is to be used on following websites as an unsupervised clustering algorithm, this being what the author describes as unsupervised.

### STAVIES
It starts from the hypothesis that data resides in leaf nodes which have a repetitive structure. The leafs are clustured and the cluster with the least statistical variance is chosen.

### VIPS
Based on the assumption that there are visual cuese(namely borders, font sizes, colors, font sizes/families, weights. Visual features are added upon DOM nodes.

A tree is built from these based on a set of 12 heuristics. A lot of them involve separator detection and ow similar blocks are.

### VIDRE
Uses VIPS for creating a tree. Selects the data region based on heuristics.

## Conclusion
Overall, the survey presents a lot of different region extractors, which mostly use heuristics and non-ML methods to extract regions of data.

```python

```
