#!/bin/bash

# split seed andn umber of workers
NUM_WORKERS=$2
DATADIR=$1

cd $DATADIR/..
mkdir $DATADIR/{external,raw,interim,final} $DATADIR/interim/{dragnet,cleaneval} $DATADIR/raw/{cleaneval,dragnet} $DATADIR/final/{dragnet,cleaneval}
mkdir $DATADIR/partd

cd $DATADIR/external/

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

rm -rf dragnet_data


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
python -m learnhtml.cli.utils convert --blocks --num-workers $NUM_WORKERS --cleaneval external/cleaneval raw/cleaneval

echo DRAGNET
python -m learnhtml.cli.utils convert --blocks --num-workers $NUM_WORKERS --dragnet external/dragnet raw/dragnet

# extract features
echo EXTRACTING FEATURES
echo CLEANEVAL
learnhtml dom --num-workers $NUM_WORKERS raw/cleaneval/raw.csv interim/cleaneval/feats-\*.csv

echo DRAGNET
learnhtml dom --num-workers $NUM_WORKERS raw/dragnet/raw.csv interim/dragnet/feats-\*.csv


# merge data but do not split afterwards
echo MERGING CSVS
python -m learnhtml.cli.utils merge --cache ./partd --on "url,path" dragnet-\*.csv interim/cleaneval/feats-\*.csv raw/cleaneval/labels.csv
python -m learnhtml.cli.utils merge --cache ./partd --on "url,path" cleaneval-\*.csv interim/dragnet/feats-\*.csv raw/dragnet/labels.csv

rm -r interim external raw partd
