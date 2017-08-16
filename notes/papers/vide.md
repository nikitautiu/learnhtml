## ViDE: A Vision-based Approach for Deep Web Data extraction

*ViDE: A Vision-based Approach for Deep Web Data Extraction, Wei Liu et al. 2007*

### Introduction
* *Data record* and *data item* extraction based on engineered visual features
* Referrs especially to unindexed, unstrctured pages
* Assumption of visual regularity
* Also uses some non-visual features(data type) for robustness

Employed steps:
1. Obtain a visual block tree from the web page
2. Extract data records from it
3. Extract data items from it and align the semantically similar ones
4. Generate visual wrappers

**Note** Extracts visual infomration heuristically from HTML, not render time.


### Related works

* Uses **revision** to measure performace(the number of web pages whose data records cannot be perfectly extracted. Precision and recall are not both 100%).
* Based on the **VIPS** algorithm which is also used elsewhere in literature.
* **VENTex** uses CSS render boxes, but not to extract semantic information just tables.


### Visual features
* position (x, y) on page
* Font features(size, face, color, decoration, weight, frame)

Visual tree. Leaves are elements that cannot be segmented further, representing sematic units.

#### Position features
Heuristically assumes the data region is centered horizontally and it's the largest of the page. The data region is greater or equal to 0.4 of the page.

#### Layout features
Assumes grid or vertical list layout for data records. Flush left. Data does not overlap and is equally spaced

#### Appearance features
Used for visual features inside data records. Data records and data items with the same semantics have similar appearance. Usually uses distinguishable fonts.

**Note** they use both absolute and relative position

#### Content features
Related to the content, as data records correspond to cohesive structured data. They ar eclassified in *mandatory* and *optional*
* CF1: The first data item in each data record is always of a mandatory type.
* CF2: The presentation of data items in data records follows a fixed order.
* CF3: There are often some fixed static texts in data record


#### Data record extraction
Depends on finding the data region with the heuristics described above. An ideal extrtor has the following 2 requirements:
1. all data records from the region are extracted
2. all data items from within a record are extracted

Each data record corresponds to one or more sub-trees in the visual tree. *Noise blocks* used to reffer to either top or bottom visual blocks which do not convey information.

### Performance
Good precision and recall, mediocre revision compared to MDR for data record and DEPTA for data item.
