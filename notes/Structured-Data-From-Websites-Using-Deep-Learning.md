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
* 2 . Classifying HDL based on visual features.
* 3 . Leave-one-out prediction for feaures to see corelations and decide which is stronger

We will be splitting the data into 3:
* D1. data on pages containing the labels
* D2. data on pages of the same website
* D3. data on all the websites

### Results
* R1. 1A vs 1Brand vs 2
* R2. 1A & 2 for HDL

### Usecases
* API generation from html websites - Given a website, extract structured data and expose it as an API
  * for a single unindexed site
  * for multiple similar websites(ie. blogpost aggregator)
* borad crawls for complex structured data - Broad crawls are notoriusly hard to do if the data is not immediatley parsable. A model that cand extract random structured data could accomplish such a task. ( this depends on how well the model generalizes)
* site migration. After extracting structured data, the site could be migrated to another html structure.

### Conclusion

### Related work
* *Cross-Supervised Synthesis of Web-Crawlers - Adi Omari* 
  * Generating XPaths from labels
  * Generalizing to new websites based on equivalence relationship between content(with edit distance). This is based on the assumption that the same exact data is on both websites.
  * Generating XPaths for the new site
  * Three-tiered filtering of content. Selectign urls, selectign data container and selcting data from the container
  
  This could be relevant as the process of gathering new data with the equivalence relationship can help groe the dataset synthetically.

### Other ideas
1. Decide framework of website from features
2. Decide content type(blog, forum, ecommerce, tutorial, etc)
3. Extract semantic tree information from website(needs more data)
4. Try using RNNs on DOm trees
