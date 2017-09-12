# Boilerplate Detection using Shallow Text Features - Kohlschutter et al. 2010

## Introduction
Presents a *boilerplate detection* algorithm using shallow text features. They test its performance, and also its impact on the performance of  it as  preprocesssing step in content extraction.

Only uses 2 features: **number of words** and **link density** with a simple classification model. Does two types of experiments: 
* two class - `noise/no-noise`
* domain specific 4 class classification(news sites)

## Related works
Compares aother works that evaluate agiant the **CleanEval** benchmark.

## Web page features
Notes that n-gram models tend to grow to very large sizes. The paper avoids toke-based apporaches altoghether. Describes 4 levels of feature extraction:
* individual tags
* entire HTML content
* rendered document image
* entire website

Notes that the latter two require CSS. Notes that it may bebeneficial to add data from the two external levels, but it's also computationally expensive. 

Notes that tag becomes less relevant as CSS becomes more prevalent. To avoid overfitting to one particular site, they use only take into consideration a very small subset of tags. 

### Shallow text features
Uses **quantitative linguistics** features such as average word length, average sentence length and absolute number of words. Build upon this with a few heuristics. It argues that a word/token oriented approach would overfit to a particular domain of websites. Adds a few feuristics such as number of words starting with an uppercase, number of vertical bars, etc.

Other features include densitometric features such as the **link density**(which is the ration of  links tokens to text tokens). IT also adds **text density**(which is the number of words for every line of a block, assuming a particular text warp).

## Classification
The web page is segmented into blocks of text annotated with the above features. It uses two sets of labels as described above. The domain specific tags in this case are:
* headline 
* comments
* related content
* supplemental
* full-article text

The data is classified using **decision trees** and **SVMs**.

## Evaluation
The dataset is **available for download** for research purposes. It also tests against the **CleanEval** dataset. The algorithm achieves 90%+ f1-scores on the GoogleNews dataset. IT achieves higher than baseline perfrmance on the CleanEval dataset, but lower than on the other. 

## Exploratory analysis
The paper proposes a quantitative liguistic motivation to the text density vs number of words performance.

```python

```
