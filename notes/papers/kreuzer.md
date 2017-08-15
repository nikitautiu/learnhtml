## A quantitative comparison of semantic web page segmentation algorithms - Kreuzer 2013

### Ontlogy of extraction methods
1. DOM based - use oly html. Makes the claim that they may not convey enough information
2. Visual based 
3. Text based - text density, link density. based on quantitatie linguistics

### Related work
References *Identifying Primary Content from Web Pages and its Application to Web Search Ranking*, which groups tags into content segments or **noise segments**. Uses visual and content properties.

*Extracting content structure for web pages based on visual representation*

*Browsing on Small Screens: Recasting Web-Page Segmentation into an
Efficient Machine Learning Framework* - usese **DOM + visual features** to classify mobile web pages using decision trees to segment a page into 9 areas.

*A segmentation method for web page analysis using shrinking and dividing* uses an edge detection algorith, on a page to segment it.

Most writers make their dataset by hand-labeling pages into semantic blocks. Sometimes data is labeled **both by authors and volunteers**, to avoid bias.


### Approach
*Semantic block* = contiuous html fragemnt which renders as graphically consistent block. 

Segmentation with granularity of 1 => flat segmentation.

Blocks are usually visually defined but do not have direct dom correspondence, as many nested blocks can reside one in the other. This **granularity** has to be specified.

Chains of nested dom elements can be **grouped together** as there is no one mapping telling which one is the relevant one.


Uses *recall*, *precission* and *F1score*.

For evaulation tests both exact and *fuzzy* matching. For exact matching, it checks wehther the results are exactly the ones expected. For fuzzy, it checks whether the result matches the expection with 80% accuracy.

### Algorithms
* *BlockFusion* - quanitative textual data
* *PageSegmenter* - dom-based segmenter basend on root-to-leaf path similarities (ie. `/html/body/p/ul/li`)
* *VIPS* - rendered page elements, with visual information. Edge detection etc. heuristic
* *WebTerain*

WebTerrain uses heuristics based on the depth of the dom tree, and the positions and surface are to choose *dominant* tags. Expermiantal results show good performance with these features.

```python

```
