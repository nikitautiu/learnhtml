# A hybrid approach for content extraction with text density and visual importance of DOM nodes - Song et al. 2012

## Introduction
Proposes a method of discerning noise from actual content of the page. This would be considered a region extractor in similar literature. Other works estimate noise at 40-50% of web content in 2005, continuing to increase.

Based on the assumtion that decorations are usually fairly structured and contains brief sentences and content is usually in a simple format and contains a lot of hyperlinks. Uses **text density** and **omposite text density** as metrics for text of dom node. Also, uses  **visual importance** for each node.

## Related work
Describes a series of other region extractors. Some of them are **template detection** algorithm which extrapolate what is noise and what is not from the similarities of different pages.

It references works suck as Denbath et al.'s FeatureExtractor algorithm which analyzes blocks for tag content and amount of text.  Gottron developed CombineE and Peters and Lecocq use ensamble and **machine learning** methods wwith different features to extract data.

## Method
Extracts from the dom tree for each node the **text density**. Text density is the ration of character to tags in a subtree. It is large for nodes containing simply formated text and low for highly formatted nodes.

**Composite text density** is described as in terms of link char numbers and link tag numbers. It also has some bakancing terms to normalize too extreme values in case of long texts and  highly formated series of links. 

**Visual importance** is defined in term of horizontal alignment and relative size of a node. 

It then uses a metric ocmposed of all the above metrics to determine a threshold of importance for which nodes become content.
