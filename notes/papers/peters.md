# Content Extraction Using Diverse Feature Sets - Peters et al. 2013

## Introduction
Aims to separate main content fr om navigation and eye-candy via a diverse feature set. Also extracts semantic information uvia `id` and `class` attributes. It describes its elf as a content extraction algorithm. In terms of ontology, this algorithm is used for `noise/content` classficiation and would rather be a **region extractor**.

References the works of  Kohlschutter and the CETR algorithm as application of machine learning in of region extraction. It elaborates on those two and adds **semantic information from the id and class attributes**.

## Approach
It only takes into account tags containing text. It then searches in the dataset for the text using fuzzy text matching to determine what tags are to be labeled as content. The code can be found at http://github.com/seomoz/dragnet AS for the model, it uses regularized logistic regression.

The dataset used to benchmark is taken from the CETR algorithmm.(this could be very useful for using with our own approach, relieves us of having to build a dataset). The data is from 2012

### Model features
It uses text and link density inspirde by Kohlschutter's work. **Tag ratio** is also used(text length to number of tags). It also uses the output of CETR as a feature of the model.

As far as semantic features go, the `id` and `class` are tokenized, and only those with a  content-to-no-content ratio greater tha 2.5 that appear in more tha 1% of tags are selected. The total number of tokens ammounts to 8 per id and 24 per class.

## Results
The results are better than CETR and yield a F1 score of roughtly .85. The **semantic features** alone yield failry high scores, while combined with the **Shallow Text features**(tag density) they reach the final performance. The semantic features are more relevant on the more recent 2012 dataset, given the rising popularity of CSS.

```python

```
