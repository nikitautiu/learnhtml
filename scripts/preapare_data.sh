#!/bin/bash

# split seed andn umber of workers
SPLIT_SEED=42
NUM_WORKERS=16

cd ..
mkdir data data/{external,raw,interim,final} data/interim/{dragnet,cleaneval} data/raw/{cleaneval,dragnet} data/final/{dragnet,cleaneval}
mkdir ~/partd

cd data/external/

# clone and untar
git clone https://github.com/seomoz/dragnet_data.git
cd dragnet_data
tar xvf dragnet_HTML.tar.gz
tar xvf dragnet_Corrected.tar.gz

# copy the cetr data as well
wget https://www3.nd.edu/~tweninge/cetr/cetr-dataset.zip
unzip cetr-dataset.zip -d cetr-dataset
chmod +x cetr_to_dragnet.sh
./cetr_to_dragnet.sh cetr-dataset > /dev/null

# move it to the other directory and remove the junk
cd ..
mkdir dragnet
mkdir cleaneval
mv -t dragnet dragnet_data/{HTML,Corrected}
mv -t cleaneval dragnet_data/cetr-dataset/cleaneval/en/{Corrected,HTML}

rm -r dragnet_data


# recode data
# convert both txt and html files
for f in $(find . -name "*.txt" -o -name "*.html"); do
        encoding=$(file -i $f | cut -d"=" -f 2)  # get the mime encoding
        if [ "$encoding" != "us-ascii" ] && [ "$encoding" != "utf-8" ]; then
                res=$(chardetect $f)  # try to detect it otherwise
                encoding=$(echo $res | cut -d" " -f 2)
                echo $res - CONVERTING TO UTF-8
                recode ${encoding}..utf-8 $f
        fi
done

# remove the unsolvable ones
cd dragnet
rm HTML/{R121,T19,T2,T31}.html Corrected/{R121,T19,T2,T31}.html.corrected.txt
cd ../cleaneval
rm HTML/{114,276,305,376,767,331,619,716}.html Corrected/{114,276,305,376,767,331,619,716}.html.corrected.txt


echo EXTRACTING LABELS/RAW
cd ../..
echo CLEANEVAL
../src/cli.py convert --num-workers $NUM_WORKERS --cleaneval external/cleaneval raw/cleaneval

echo DRAGNET
../src/cli.py convert --num-workers $NUM_WORKERS --dragnet external/dragnet raw/dragnet

# extract features
echo EXTRACTING FEATURES
echo CLEANEVAL
../src/cli.py dom --num-workers $NUM_WORKERS raw/cleaneval/raw.csv interim/cleaneval

echo DRAGNET
../src/cli.py dom --num-workers $NUM_WORKERS raw/dragnet/raw.csv interim/dragnet


# merge data
echo MERGING CSVS
../src/cli.py merge --cache ~/partd --on "url,path" interim/cleaneval/dom-full-\*.csv interim/cleaneval/feats-\*.csv interim/cleaneval/oh-\*.csv interim/cleaneval/freqs-\*.csv raw/cleaneval/labels.csv
../src/cli.py merge --cache ~/partd --on "url,path" interim/dragnet/dom-full-\*.csv interim/dragnet/feats-\*.csv interim/dragnet/oh-\*.csv interim/dragnet/freqs-\*.csv raw/dragnet/labels.csv

# split the data
echo TRAIN/VALIDATON/TEST SPLIT
../src/cli.py split --state $SPLIT_SEED --on url interim/cleaneval/dom-full-\*.csv final/cleaneval/dom-full-train-\*.csv 75 final/cleaneval/dom-full-validation-\*.csv 15 final/cleaneval/dom-full-test-\*.csv 15
../src/cli.py split --state $SPLIT_SEED --on url interim/dragnet/dom-full-\*.csv final/dragnet/dom-full-train-\*.csv 75 final/dragnet/dom-full-validation-\*.csv 15 final/dragnet/dom-full-test-\*.csv 15