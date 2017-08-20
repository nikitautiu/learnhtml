## Learning deep structured semantic models for web search using clickthrough data - Huang et al. 2013

* wants to map contant to a lower dimensional semantic space
* tries to do things similar to probabilistic LSA and Latent Dirichlet Allocation - but unlike those(which are unsupervised) does it in a supervised manner with clicktrhough data(query->wether it is clicked)
* Hinton uses **deep-autoencoders** to extract semantics from the textual data. bu doesn't explore other metrics other than reproduction -- referenced by a document


### Deep learning 
#### Related work
* uses the cosine-similarity metric as a loss in training the model
* Hinton uses an autencoder on the term vector extracted from the content and the query(the term vector is extracted using a restricted Boltzmann machine) - the resulting bottlenecks are compared to rank similarity between **query and content** - **Semantic Hashing**
* they use only the first 2000 most common words

#### This work
* uses a **deep-autoencoder** where samples are actually query term vector concancatenated with the document term vectors. each vector is reduced in size with its **independent** hidden layers
* the pre-readout layer is one that calculates cosine similarity between query and the others and the predicted values is the softmax probability representing clcikthrough percentage
* uses word hases to reduce words to n-grams
