## Extracting Structured Data from Websites Using Deep Learning

### Abstract(what we want to achieve?)
The goal of this document is to explore the posibility of extracting manually-labeled structured data records from websites, using deep learning models. Such a model would be able, given a set of websites with the tags containing the desired information labeled accordingly, to both extract data from the same websites and potentially generalize to other similarly-structered sites.

### Introduction(what do we do?)
We tackle the following tasks:
1. Classification based on **DOM**-tree features. We are extracting features strictly related to the dom, for each tag. Fo each tag, we are also adding features based on its predecessors and descendants. (See notebook `1-html-features`.)

2. Classification based on **redender boxes** and other visual features.

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

### Conclusion

### Related work
Other papers and how they ralate to this(~3 papers)

### Other ideas
1. Decide framework of website from features
2. Decide content type(blog, forum, ecommerce, tutorial, etc)
3. Extract semantic tree information from website(needs more data)
