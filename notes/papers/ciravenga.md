## Learning to Harvest Information for the Semantic Web - Ciravenga et al. 2004

* does not attempt to discover raltionshps between entities
* the method **generalizezs** and is **unsupervised**
* **domain-specific annotation** - as in domain of interesnt not web domain
* begins from a give **lexicon** and asks for user confirmation when seraching for data - sort of **active learning**
* based on the assumption of **redundancy** - ie. learning what info should be extracted from other sources by its appearance in the first one


* data is stored as triples of subject-verb-object 
* confidence in the guess is done throught number of referencing sources - ie how many 
* it is used for **very specific tasks** - not general semantic segmentation, but rather very specialized tasks depending on reduntant information to be present -eg finding all papers of a ceratin author


* it presents more of a framework for implementing such problems
* **doesn't use machine learning** - but presents ML as a possible imporvement to the current method for reducing the search space
