## LearnHtml

Html web content extraction library using *mostly* DOM features as well as some textual features. [Achieves](https://www.researchgate.net/publication/329061153_Learning_Web_Content_Extraction_with_DOM_Features) a tag-level F1-score of `.96` on the Dragnet dataset.

### Requirements
First you will need to install the dependencies. For the binary dependencies:

```bash
sudo apt-get install recode libxml2-dev libxslt1-dev unzip
```

Python dependencies:
```bash
pip install -r requirements.txt
```

Build the project and install it locally
```bash
pip install -e .
```

### Running the scripts

```bash
./learnhtml/cli/prepare_data.sh <<WHERE_TO_DOWNLOAD_DATA>> <<NUMBER_OF_WORKERS>>
```

Copyright (C) 2018 Nichita Uțiu
