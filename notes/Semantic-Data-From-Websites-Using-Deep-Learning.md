## Extracting Semantic Data from Websites Using Deep Learning

### Abstract(what we want to achieve?)
The goal of this document is to explore the posibility of extracting manually-labeled structured data records from websites, using deep learning models. Such a model would be able, given a set of websites with the tags containing the desired information labeled accordingly, to both extract data from the same websites and potentially generalize to other similarly-structered sites.

### Introduction(what do we do?)
We tackle the following tasks:
1. Classification based on **DOM**-tree features. We are extracting features strictly related to the dom, for each tag. Fo each tag, we are also adding features based on its predecessors and descendants. (See notebook [`1-html-features`](../experiments/1-html-features.ipynb).)
2. Classification based on **redender boxes** and other visual features.
    * possibly use both `border-` and `content-` boxes
    * use both absolute and relative positioning
    * **CANNOT** be directly mapped between *human visually sepaprable blocks* to tags, so the classification must be done more relaxed or grouped somehow. **TODO** decide on a criterion of grouping, either learnable(preferable) or heuristic

### Experiments(how we aproach the problem?)
For the the first set of features, we will be trying to classify both on the HLD(hand labeled data) and on some syntheticly generated labels.

* 1A. Classifying using only DOM  fetures on HDL
* 1B. Classifying on randomly generated XPath labels(generating labels for tags extracted with the same random Xpath) to see how does the ML model compare.(**note** this is not useful for semantic segmentation, jsut to see the overall capablities of the model)
* 2. Classifying HDL based on visual features.

#### Experimental design
As for how we will be doing dataset splitting for experiments, we will be iterating on the the 3 datasets proposed in [5-data-preparation](5-data-preparation.ipynb) and on the datasets proposed by [Burget et al.](../notes/papers/burget.md). We we also justify this decision through a mathematical assumption. 

We assume that webpages within a site are implementaed based on several types of templates which are resused within it. Every tag on a page's extracted features can be considered a random variable $T_{s,t}$ distributed according to site $s$'s template $t$'s distribution. For a machine learning, or any kind of statistical estimator to perform well the samples must accurately portray the distribution. These distributions are purely hypothetical, so we do not know how much they diverge between templates of the same site or between those of different sites. 

To test this, we will split the date as mentioned before: 
1. data on pages containing the labels - $\{ T_{s,t} \mid \text{pages t contain label l} \} \text{where l and s fixed}$
2. data on pages of the same website - $\{ T_{s,t} \} \text{where l and s fixed}$
3. data on all the websites -  $\{ T_{s,t} \} \text{where l fixed}$

This means $ 1 \subset 2 \subset 3 $


For these datsets, we will have 5 experiments. To check the divergence between the distributions and the model's generalization capability, we will have two types of test scenarios:
1. Training done on dataset $i$ and test still on $i$
2. Training done on $i$ and test on $i+1$

All but the last dataset will have the second variation of testing, as tere is no more data to test on for it. The motivation of the second experiment is to see the potential of  actually **reducing the size of the training dataset**(maybe even for potential future work using online learning).

### Results
* R1. 1A vs 1Brand vs 2
* R2. 1A & 2 for HDL

### Related work
#### [Web Page Element Classification Based on Visual Features - Burget et al. 2009](papers/burget.md)
* Uses **ML** - decision trees for classification of segmented visual blocks
* The **visual features are extracted with basic CSS rendering** 
* Uses 5 classes of fetures that overlap with what we are trying to do:
    * Size and position. x,y, width, height
    * background colors 
    * font properties
    * border properties
    * Text features(total digits, total lower, total upper, lenght etc)
* Uses heuristic line and block detection which may be achievable with a CNN
* heuristically defines **visually separably** blocks in a tree. **Leafs are content**(text, images) and the rest are DOM nodes that are deemed **visually separate**. The tree is fed to VIPS. => This step could be implemented using a CNN that is trained to identify visual areas(**NOT dom tag** - as there is no one mapping from nodes to visual blocks - see *Kreuzer*)
* Similar **experimental design**:
    1. trains and tests on the same site
    2. tries to test generalization by testing on sites others than those used in training
* Similar labels: groups image segments into groups specific to the domain of the problem(articles -> title, author, content, etc) - **flat segmentation**

#### [ViDE: A Vision-based Approach for Deep Web Data extraction - Liu et al. 2007](papers/vide.md)
* assumes visual regularity among semantically similiar items
* font and positional features
* searches for gird/list patterns in the page(doable with CNN?)
* quanitifes results using **revision** - percentage of pages with $F1_{score} = 1$ => $F1_{score}$ may be used as a more relaxed performance metric
* **three-leveled** extraction: data region - data records - data items. heirarchy of more granular groups of data => could be useful as an idea, but it is a a pretty string assumption 
* has a visual segmentation step that creates a **visual tree** based on(boxes, font, image, hyperlink content etc.) These are then grouped using heuristic observation 
* basically does **three level segementation** => granularity = 3 (data region - data record - data item)
* **idea for future work** => arbitrary granularity visual block segmentation

#### [A quantitative comparison of semantic web page segmentation algorithms - Kreuzer 2013](papers/reuzer.md)
* many papers describe **two-step process** of segentation and classification => both our steps will be ML, and the first is not a segmentation per-se but a visual classification step, as the segmentation is not general purpose, but domain-specific
* defines a **semantic block** as continuuous HTML fragment rendered as a graphically consistent block
* defines **granularity** as the nesting level of blocks. in this case we are basically doing flat segmentation or a basic 2 level one
* **no single mapping between visual block** and html block. A heuristic such as **grouping  toghether** adjacent tags that are part of the same semantic block  => We should be also be searching for the **block** that contains the semantic info necessary at the visual step, then enance it with other features. The visual step should detect all tags within the group

* some sort of fuzzy matching metric could be defined, maybe based on the above definition of semantic blocks and the use of grouping to relax the constraints
* mentions good performance of **depth, positioning and area** in the WebTerrain segmentation algorithm
* there are **no well-established datasets** avaialble to compare perfomance against => free to use whatever

#### [Semantic Partitioning of Web Pages - Vadrevu et al. 2005](papers/vadrevu.md)
* uses information theroy **entropy** to segemnt a the page based on **root-to-leaf** paths => our neighbourhood could be seen as a particularization of the concept
* it is basically a **clustering** algorithm that doesn not make assumptions about the template of the page such as other methods => our mathod is supervizd,but aims to infer any sort of underlying template structure
* similar to the *ViDE* region-record-item heirarachy
* we might try to minimize such entropy as well with a clustering algorithm on the leaf nodes(potential future work??)

### Conclusion



### Other ideas
1. Decide framework of website from features
2. Decide content type(blog, forum, ecommerce, tutorial, etc)
3. Extract semantic tree information from website(needs more data) - **semantic page segmentation**
4. Researcha active learning and somehow try to obtain more and more data for the dataset
