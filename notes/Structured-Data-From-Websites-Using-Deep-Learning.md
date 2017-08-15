## Extracting Structured Data from Websites Using Deep Learning

### Abstract(what we want to achieve?)
The goal of this document is to explore the posibility of extracting manually-labeled structured data records from websites, using deep learning models. Such a model would be able, given a set of websites with the tags containing the desired information labeled accordingly, to both extract data from the same websites and potentially generalize to other similarly-structered sites.

### Introduction(what do we do?)
We tackle the following tasks:
1. Classification based on **DOM**-tree features. We are extracting features strictly related to the dom, for each tag. Fo each tag, we are also adding features based on its predecessors and descendants. (See notebook [`1-html-features`](../experiments/1-html-features.ipynb).)

2. Classification based on **redender boxes** and other visual features.
    * possibly use both `border-` and `content-` boxes
    * use both absolute and relative positioning

### Experiments(how we aproach the problem?)
For the the first set of features, we will be trying to classify both on the HLD(hand labeled data) and on some syntheticly generated labels.

* 1A. Classifying using only DOM  fetures on HDL
* 1B. Classifying on randomly generated XPath labels(generating labels for tags extracted with the same random Xpath) to see how does the ML model compare.
* 2. Classifying HDL based on visual features.

We will be splitting the data into 3:
* D1. data on pages containing the labels
* D2. data on pages of the same website
* D3. data on all the websites

### Results
* R1. 1A vs 1Brand vs 2
* R2. 1A & 2 for HDL

### Related work
#### [Web Page Element Classification Based on Visual Features - Burget et al. 2009](papers/burget.md)
* Uses machine learning - decision trees for classification of segmented data
* Uses 5 classes of fetures that overlap with what we are trying to do:
    * Size and position. x,y, width, height
    * background colors 
    * font properties
    * border properties
    * Text features(total digits, total lower, total upper, lenght etc)
* Uses heuristic line and block detection which may be achievable with a CNN
* Similar **experimental design**:
    1. trains and tests on the same site
    2. tries to test generalization by testing on sites others than those used in training
* Similar labels: groups image segments into groups specific to the domain of the problem(articles -> title, author, content, etc)
    
#### [ViDE: A Vision-based Approach for Deep Web Data extraction - Liu et al. 2007](papers/vide.md)
* assumes visual regularity among semantically similiar items
* font and positional features
* searches for gird/list patterns in the page(doable with CNN?)
* quanitifes results using **revision** - percentage of pages with $F1_{score} = 1$ => $F1_{score}$ may be used as a more relaxed performance metric
* **three-leveled** extraction: data region - data records - data items. heirarchy of more granular groups of data => could be useful as an idea, but it is a a pretty string assumption


#### [A quantitative comparison of semantic web page segmentation algorithms - Kreuzer 2013](papers/reuzer.md)
* defines a **semantic block** as continuuous HTML fragment rendered as a graphically consistent block
* defines **granularity** as the nesting level of blocks. in this case we are basically doing flat segmentation or a basic 2 level one
* no direct correspondence between visual block and html block. A heuristic such as **grouping  toghether** adjacent tags that are part of the same semantic block
* some sort of fuzzy matching metric could be defined, maybe based on the above definition of semantic blocks and the use of grouping to relax the constraints
* mentions good performance of depth, positioning and area in the WebTerrain algorithm

### Conclusion



### Other ideas
1. Decide framework of website from features
2. Decide content type(blog, forum, ecommerce, tutorial, etc)
3. Extract semantic tree information from website(needs more data)
