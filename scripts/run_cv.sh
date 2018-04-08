#!/bin/bash

die () {
    echo >&2 "$@"
    exit 1
}

RANDOM_SEED=42
RESULTS_PATH="../experiments/results/"
DRAGNET_PATH="../data/final/dragnet/dom-full-\*.csv"
CLEANEVAL_PATH="../data/final/cleaneval/dom-full-\*.csv"
ALL_FEATURES=(numeric text both)
ALL_ESTIMATORS=(logistic svm tree random deep)

# param validation
[ "$#" -le 3 -a "$#" -ge 1 ] || die "between 1 and 3 arguments required, $# provided"

DATASET=$1
ESTIMATORS=${2:-${ALL_ESTIMATORS[@]}}
FEATURES=${3:-${ALL_FEATURES[@]}}

if  [ "$#" -ge 1 ]; then
    [ $DATASET == "dragnet" ] || [ $DATASET == "cleaneval" ] || die "dataset must be either cleaneval or dragnet"
fi

if [ "$#" -ge 2 ]; then
    [ $ESTIMATORS == "logistic" ] || [ $ESTIMATORS == "svm" ] || [ $ESTIMATORS == "tree" ] || [ $ESTIMATORS == "random" ] || [ $ESTIMATORS == "deep" ] || die "invalid estimator"
fi

if [ "$#" -ge 3 ]; then
    [ $FEATURES == "numeric" || $FEATURES =="text" || $FEATURES == "both" ]  || die "invalid features"
fi


# set dataset file
DATASET_PATH=$CLEANEVAL_PATH && [[ $DATASET == "dragnet" ]] && DATAESET_PATH=$DRAGNET_PATH

# ..src/cli.py evaluate --estimator logistic --features numeric --external-folds 10 10 --internal-folds 5 5 --n-iter 1 ../data/final/cleaneval/dom-full-\*.csv ../experiments/results/cleaneval-logistic-numeric-cv.pickle

# TODO: solve the file problem, don't just print the script!

for ESTIMATOR in $ESTIMATORS; do
    for FEATURE in $FEATURES; do
        FILENAME="${DATASET}-${ESTIMATOR}-${FEATURE}"
        RESULT_FILE="${RESULTS_PATH}${FILENAME}.pickle"
        LOG_FILE="${RESULTS_PATH}${FILENAME}.log"
        N_JOBS="" && [[ $ESTIMATOR == "deep" ]] && N_JOBS="--n-jobs 1"
        
        # run the command
        echo "../src/cli.py evaluate --estimator ${ESTIMATOR} --features ${FEATURE} --external-folds 1 10 --internal-folds 10 10 --n-iter 60 ${N_JOBS} ${DATASET_PATH} ${RESULT_FILE} > ${LOG_FILE} &"
    done 
    
    echo "wait"
done