## Extracting Semantic Data from Websites Using Deep Learning

### Abstract(what we want to achieve?)
The goal of this document is to explore the posibility of extracting manually-labeled structured data records from websites, using deep learning models. Such a model would be able, given a set of websites with the tags containing the desired information labeled accordingly, to both extract data from the same websites and potentially generalize to other similarly-structered sites.

### Introduction(what do we do?)
Whereas a lot of the papers explored make use of heuristics to classify content even when using ML models(**TODO** reference them), for the purpose of this experiment we will explore the ability of a deep learning model to discover features. We will be making the conscious decision of not making any algorithmic design decision based on heuristics and keeping heuristicaly significatnt features to a minimum, possibly using none at all. This way we will not make any assumptions on the structure of the data, and hope any signal that arises will be picked up by the model.(a similar approach is taken by *Peters* in his paper)

We will be using th efollowing set of features:
1.  **DOM**-tree features. We are extracting features strictly related to the dom, for each tag. For each tag, we are also adding features based on its predecessors and descendants. (See notebook [`1-html-features`](../experiments/1-html-features.ipynb).) Thsi subset does not make any assumption on the textual content of the tag
2. **Visual features**
    * possibly use both `border-` and `content-` boxes
    * use both absolute and relative positioning
    * **CANNOT** be directly mapped between *human visually sepaprable blocks* to tags, so the classification must be done more relaxed or grouped somehow. **TODO** decide on a criterion of grouping, either learnable(preferable) or heuristic
3. **Textual features**

## Datasets
There are a lot of datasets used to benchmark content extraction(CE) in literature, however, those that are used for more than one paper, let alone publicly available are few and far between. This section serves to  iterate a few of them as a reference and to choose a good one for our experiments.

### CleanEval
* released in 2007 for the CleanEval competition
* 741 english pages, 713 chinese pages
* **2-class** - content/no-content
* used alongside a benchmark which is based on Levenshtein editing distance - not necessary
* available in two flavour **CleanEval-EN** and **CleanEval-ZH** which have respectively, english and chinese pages
* contains a wide variety of pages

### MSS
* made in 2009 by Pasternack and Roth for their paper
* contains pages from 45 websites - split into **Big5** and **Myriad40** - 450 in total
* **2-class**
* contains news sites

### L3S-GN1
* used by Kohlschutter in 2010
* both **2-class** and **4-class**, the latter being domain specificto news websites
* 621 pages from 408 websites
* named **GoogleNews** in the paper


### CETR
* a combination of **CleanEval** and **MSS**
* MSS is sanitized and better formated 

### Dragnet
* from 2012
* **2-class**
* contains
    * 999 pages from randomly selected RSS feeds
    * 204 pages from 23 news websites 
    * 178 pages from a blog directory
* includes data from **CleanEval** and **CETR**

### TECO
* from 2016
* **2-class** 
* preserves **presentational information**(pages are downloaded with CSS and JS)
* 50 pages, 1 from each website are labeled 
* **per-tag labeling** as opposed to text-block as in the others - not a big issue
* the sites are diverse in topics

### Conclusion
For example Yao uses in his paper both L3S-GN1 and Dragnet, which could probably make for the most complete dataset. Most of them use text blocks in the definition of Kohlschutter, but tags can be retrieved via text blocks as there is a 1-to-1 mapping.

### Experiments(how we aproach the problem?)
For the the first set of features, we will be trying to classify om ur dataset and the benchmark ones, each time using subsets of the following sets of features

* 1. DOM features(**DOM**)
* 2. Visual features(**VIS**)
* 3. Textual features(**TEX**)

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

**Note** Behind this motivation also stands the fact that almost all of the data is labelable and was labeled using xpaths, which means there is some sort of consistent pattern within the pages of the website.

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
* some sort of fuzzy matching metric could be defined, maybe based on the above definition of semantic blocks and the use of grouping to relax the constraints
* mentions good performance of **depth, positioning and area** in the WebTerrain segmentation algorithm
* there are **no well-established datasets** avaialble to compare perfomance against => free to use whatever
* many papers describe **two-step process** of segentation and classification => both our steps will be ML, and the first is not a segmentation per-se but a visual classification step, as the segmentation is not general purpose, but domain-specific
* defines a **semantic block** as continuuous HTML fragment rendered as a graphically consistent block
* defines **granularity** as the nesting level of blocks. in this case we are basically doing flat segmentation or a basic 2 level one
* **no single mapping between visual block** and html block. A heuristic such as **grouping  toghether** adjacent tags that are part of the same semantic block  => We should be also be searching for the **block** that contains the semantic info necessary at the visual step, then enance it with other features. The visual step should detect all tags within the group

#### [Semantic Partitioning of Web Pages - Vadrevu et al. 2005](papers/vadrevu.md)
* uses information theroy **entropy** to segemnt a the page based on **root-to-leaf** paths => our neighbourhood could be seen as a particularization of the concept
* it is basically a **clustering** algorithm that doesn not make assumptions about the template of the page such as other methods => our mathod is supervizd,but aims to infer any sort of underlying template structure
* similar to the *ViDE* region-record-item heirarachy
* we might try to minimize such entropy as well with a clustering algorithm on the leaf nodes(potential future work??)

#### [Learning to harvest information for the semantic web - Ciravegna et al. 2004](papers/ciravenga.md)
* not very related to the subject 
* presents a framework of extracting *very* domain-specific information such as papers written by certain authors through correlation with information from other sources, and external sources of related information(names), **but** the frameowrk doesn't seem to apply to generic data  without specific tailoring of kinda *knowledge bases*

#### [Learning deep structured semantic models for web search using clickthrough data - Huang et al. 2013](papers/huang.md)
* **fascinating** and very well written article. 
* shows a new model for extracting semantic information **from text**
* elaborates on Hinton's work with **deep-autoencoders** but makes it supervised
* might be useful if used with textual data from classes and ids to learn features to feed to the model

#### [A Survey on Region Extractors - Sleiman et al. 2013](papers/sleiman.md)
* makes the distinction between **information extractors** and **region extractors** and focuses on the latter
* most of the make **strong** assumptions on the structure of the data
* some of them, namely MDR, TCP, U-REST etc. use similarity within the same html document of DOM nodes to infer regions. we are doing somethings imilar with the the **neighbourhood**
* other hypotheses are made based on the **structure of the dom tree**(depth, number of children, etc) - ex. STAVIES, TPC, MDR, OMINI
* VIPS, and VIPS-based algorithms(ViDRE, ViNTS, VSDR, RIPB) use **visual features** such as font, color, visual separators, etc. to identify relevant regions. Our method, using visual features for the regions selection. Our use of a CNN or computer vision model for classification can be seen as an extension on this process with less heuristic assumptions
* mentions **region extractors** as a sort fo preprocessing method to select relevant data. A pretrained model can be used to classify pages as containing data or not in our case, and feed the pre-readout of the page to each of its individual nodes to give it more context.

#### [Extracting Content Structure for Web Pages Based on Visual Representation - Cai et al. 2003](papers/cai.md)
#### [A hybrid approach for content extraction with text density and visual importance of DOM nodes - Song et al. 2012](papers/song-hybrid.md)
* uses a **densitometric** approach to detect what nodes are noise and which are not
* based on text density, link density and visual importance
* **text density** is defined in terms of text per tag in a subtree(number of characters) -> we should actually define the count as well
* **link density** is  similar, but for hyperlinks based on the assumption that text corpora contain a lot of text with a yperlinks in it
* **visual importance** is described as a measure of relative size in terms of horizontal placement
* the text and link density could be *discovered* by a sufficiently capable deep model coupled with the dom neighbourhood 
* we can also use the positioning with our models
* a ML model is not constrained by a **threshold** and should be able to learn what value ranges are relevant for the metrics without the need for such sofisticated normalization as in the paper

#### [Content Extraction Using Diverse Feature Sets - Peters et al. 2012](papers/peters.md)
* like Song, uses **tag and text density**(but inspired from Kohklschutter's work) as features - we can actually include less pre-processed features and leverage the power of deep models to discover them
* uses **regularized logistic regression** for classficiation
* uses an **EXISTING DATASET** and updates it -> we can use this
* extracts most frequent **tokens from `id` and `class` attributes** for semantics - what we are trying to do
* the code is **FULLY PUBLIC** - man I love this guy!
* is basically a `noise/no-noise` classifier, aka a **region extractor**

#### [Boilerplate Detection using Shallow Text Features - Kohlschutter et al. 2010](papers/kohlschutter.md)
* uses **densitometric features** such as **link density** and **text density**  and also average word length, sentence etc.
* imporves upon **number of words** and **link density** with a lot of heuristic features - we would like to avoid this
* uses the **quantitative linguistic** features for the **content text**,, notes this might lead to overfitting to the domain of the websites
* bechmarks agains its own **GoogleNews** dataset and the **CleanEval** dataset - we could expore those, both are free for downlaod
* due to the fear of overfitting, avoids using all HTML tags, but rather, only a subset of them
* in our case, we could add thse quantitativ efeatures in the textual feature subset
* mostly focuses on a lingustic data analysys in the latter part of the paper - not really relevant to us.
* uses **decision trees** and derives heuristics based on the learned rules
* includes features from **previous and following nodes** with notable performance improvement, when more context is known -- the sanme as our neighbourhood

#### [CETR - Content Extraction via Tag Ratios - Weninger et al.  2010](papers/weiniger.md)
* uses **tag ratio** as a feature(the text to tag ratio) - we can include the length of the text as a feature as well
* cites that **tag types** are NOT agood measure as the vast majority are 
`div` and `span`
* does **clustering** based n the tag ratio measure
* uses **CleanEval** and Pasternack's **MSS dataset** - we cna use both as they are modernized and available in he **Dragnet**
* makes a **tag ratio histogram** and also uses the derivatives of the smoothed version as features - states that increases from previous lines' ratios is information worth to be kept - again tags can not be analyzed self contained - *neighborhood*

#### [ A Machine Learning Approach to Webpage Content Extraction - Yao et al. 2013]([papers/yao.md)
* 2-class(content/non-content) classficiation using **SVMs**
* features 
    * uses a couple of **textual features** taken directly from Kohlschutter
    * discretizez the position of the text block on the page - **relative position**
    * includes the 10 most frequent **tokens in *id* and *class* ** from non-content
* datasets: **Dragnet** and **L3S-GN1** 
* comparable results to Dragnet

#### [Learning Block Importance Models for Web Pages - Song et al. 2004](papers/song.md)
#### [pix2code: Generating Code from a Graphical User Interface Screenshot](papers/pix2code.md)
* 2 main components:
* **CNN** for transforming raw visual input into an intermediary learned representation
* **RNN** to perform language modeling on textual description associated with the input image  
* task of **generating code** from **GUI screenshots** is similar to generating English **textual descriptions** given a **scene photography**
* 3 sub-problems:
	* Understanding the **scene**: inferring the objects, identities, positions and poses: **buttons, labels, element containers**.
	* Understanding the **text**: generating **syntactically** and **semantically correct** samples.
	* Combine scene and text(code) understanding to generate code that maps to a certain scene.

#### [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](papers/overfeat.md)
* **sliding window** (for object detection) can be implemented with a CNN.
* also **multiscale**
* model learns to predict **object boundaries**.
* **accumulation** of predicted bounding boxes
* shows that training a **convolutional network** to simultaneously **classify**, **locate** and **detect objects** in images can **boost** the **classification accuracy** and the **detection** and **localization** accuracy of all tasks.
* **avoids training on backgrounds**, thus cutting down on **training time**.

### Conclusion



### Other ideas
1. Decide framework of website from features
2. Decide content type(blog, forum, ecommerce, tutorial, etc)
3. Extract semantic tree information from website(needs more data) - **semantic page segmentation**
4. Researcha active learning and somehow try to obtain more and more data for the dataset
