#!/bin/bash

EMAIL=

python classify_svm.py gridsearch > gridsearch_out.txt
mail -s "Grid search result" $EMAIL < gridsearch_out.txt
sudo shutdown +1 "Finished grid search, shutting down"
