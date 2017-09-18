# CETR - Content Extraction via Tag Ratios - Weiniger et al. 2010

## Introduction
Wants to develop a method that is robust to change in websites, languages and styles. Makes the observation that tag type makes less of a difference in today's web design, but rather pages make heavy use of `div` and `span` and older algorithm are not adapted to dealing with such elements.

The approach extracts tag ratios and applies a couple of unsupervised clustering algorithms to determine content and non-content(one with a threshold, and a simple K-means one, then a third one is built upon the observation).

## Related work
Presents other algorithms that attempt to do this. Among them is VIPs, Crunch, It also mentions a few **template detection** algorithms as a category of their own. It removes identical content for many more documents. Mentions that this approach performs poorly on the **CleanEval** benchmark as there are few pages from the same site.

Mentions Pasternack's MSS algorithm as an example of token-based approach.

As for applications, mainly mentions the data filtration.

## Tag ratios
Mentions that tags are becoming largely irrelevant, because web developers are mainly using `div` and `span`. **Tag ratios** are the ratio of non-tag characters to the count of tags per line. They remove `script` and `style` not to skew.(here *line* means code line). Tag-less lines are considered to have 1 tag.

* we can use this if we actually include the number of characters in a tag

### Threshold method
Finds a threshold that differentiates between content and non-content. It the uses **Gaussian smoothing** adapted to a discrete function to smooth the histogram.

### K-means
K means is a lso applied on the TRs

## 2d Model 
The author argues that information about previous and following lines must be preserved, something which the last two methods do not do.

The 2d space is built from the smoothed histogram and it's derivatives, to indicate sharp modifications in the histogram. K-means with `k=3` is then applied to the resulting 2d samples.

To avoid dealing with minimified HTML, they apply a newline every 65 characters.

## Experiments
Uses the **dataset from Pasternack's MSS** and cleans it up. It then splits it up into **Big5** and **Myriad40** which are respectively specialized on news sites, or more diverse. Also tests on **CleanEval**.

Measures performance with F1-score. Mentions that CleanEval uses the Levenshtein  distance to calculate

## Results
Performs better than most algorithms and on all three of the datasets, with the exception of VIPS on the Big5 one. The author cites the smplicty as the algorithms main strength.

```python

```
