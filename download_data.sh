#!/usr/bin/env bash
#

DATA_FILE="cs5228-2310-final-project.zip"

kaggle competitions download -c cs5228-2310-final-project -p data
cd data && unzip $DATA_FILE && rm $DATA_FILE && cd ..

echo "DONE!"
