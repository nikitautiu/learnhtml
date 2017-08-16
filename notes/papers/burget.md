## Web Page Element Classification Based on Visual Features - Radek Burget et al. 2009

Based on visual blocks like ViDE. Purpose of blocks guessed based on appearance. Seeks to filter out noisy information from the web pages. It's purpose is to clean the  data that should later be used for such a task.

Assumptions:
* articles usually ahve visually differentiable titles
* there are some conventions regarding the presentation of certain content on websites given user reading patterns

Goals:
1. Identification of basic visual blocks
2. Classification of these blocks based on various features to infere meaning

Works with a model of the page rendering insteaf of modelling of document code. Detection of blocks based on algorithm, assign them to classes, agnostically of HTML.

### Related work
References Vips algorithm which is used by ViDE as well. Unlike Song, it assigns classes instead of a general importance level, it assigns classes to the elements. Looks for areas similar in their visual appearance.

### Approach
Page segmentation using line detection and subsequent bock detection. Uses a basic css box layout engine. Block detection is based on *VIPS*. Uses CSS rendering to get the features then heuristically groups contents into what they call *a basic visual area*. Creates a tree of visual areas based on whether **the eintire area of a node is contained in another**. 

Content leaf nodes(images and text boxes) are considered *visually separated*. Also if it hase a non transparent background or visible border around it. 

It uses an area grid by separating the pages into rows and columns and assigning to each visual area its starting/end row/column to describe their mutual positions. 

Uses continuous text lines to join horizontally adjacent *visually separable boxes*. It then uses VIPS to divide the found blocks. The result is a tree of segmented blocks. 


The classification step classifies the resulting visual blocks using the follosing features:

* Size and position. x,y, width, height
* number of areas to the left, right, etc.
* contrast
* font properties
* border properties
* Text features(total digits, total lower, total upper, lenght etc)

Uses rules to find "visually sepaprable" blocks. The result is a tree of areas. Feeds then the visually sepearable areas together with the features to a J48 decision tree based ib C4.5 algorithm.

Extracts classes specific to article pages. Usead an interactive annotation tool to do so. Small number of manually annotated pages.

### Experimental design
Similar to what we are trying to implement. 
1. Use the entire set as training and test
2. Use different for test and training but same source
3. 6 training sources 4 as testing, to check generalization power

```python

```
