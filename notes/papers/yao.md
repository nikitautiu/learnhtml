# A Machine Learning Approach to Webpage Content Extraction - Yao et al. 2013

## Introduction
Does **2 class**(content/not-content) classification using a **SVM**. Acknowledges text density, text-to-tag ratios etc as **heuristics** which  might not be universally applicable and might break. They do **semantic analysis** on the **class and id** attributes using **Naive Bayes** and incorporates it as a feature for the supervised learning.

## Related Work
Mentions some other works which use text-tag ratio and densities to determine which text blocks are or not boilerplate. Mentions VIPS and derivates as visual segmentation algorithms, but warns about their high computational complexity due to the need for rendering.

Template detection algorithms are mentioned, but as the author notes, their applications are limited due to the fact that they work on a single website.

The paper uses textblocks, similarly to how Kohlschutter does. It then cites similar performance to Peters' approach.

## Approach
It splits the content into text blocks. It uses **text features**, **relative position** and  **id, class token features**. Text features are taken from Kohlschutter's work, based on their Kullback-Liebler divergence derived relevance:
* number of words in block and the quotient to its previous block
* average sentence length and  block and quotient to previous block
* text density in this block and and quotient to previous block
* link density

The author also adds the relative position of the block as features, by discretizing all the positions into only M.

As for the **id and class** token features, they take all the tokens form the attributes and feed them to a Naive-Bayes model to learn the top 10 tokens indicative of **non-content** as their observations indicate these vary less than those for content


### Training
The features are then scaled and fed to a RBF SVM binary classifier. They also mentioned that tuning of gamma and C yielded little improvement and was ignored in the final evaluation.

### Dataset
Uses the **L3S-GN1** dataset used by Kohlschutter which has 621 pages from 408 different websites.(NOTE! Kohlschutter does both 2-class and 4-class, Yao only uses the binary one). It also uses the **Dragnet** dataset with 1380 webpages(dragnet includes their own 2012 data and the **CTER** dataset, which in turn contains **CleanEval-EN** and data from Pasternack's **MSS**).

### Metrics
Uses the classic precision, recall, f1-score metrics but also adapted to the more lenient *bag of words* evaluation which checks not for blocks, but for individual blocks to be classified. It also uses longest common subsequence to have a stricter metric. The baseline is Peters' method. Performance is averaged using a 5-fold split.

## Results
Results on L3S are better than on Dragnet wich indicated that Dragnet is **more complex and harder dataset**. Semantic features add a **non-trivial performance increase**. They say the performance is not perfect. Notes that some previous works make strong assumptions about the contiguity of the content text, which are strong, but applicable(we will not be using them)
