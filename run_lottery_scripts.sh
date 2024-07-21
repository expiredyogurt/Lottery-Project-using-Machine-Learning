#!/bin/bash

# Activate Anaconda environment
source /Users/kevinloke/anaconda3/bin/activate base


# Change directory to the location of the Python scripts for DMC
cd /Users/kevinloke/Desktop/Lottery\ Project/DMC/

# Run the Python script for DMC
python3 "DMC TotoLotteryV2.py"
python3 "DMC LSTM Model.py"
python3 "DMC Toto Prediction v4.py"
python3 "DMC LSTM Top 3 Prediction.py"

# Change directory to the location of the Python scripts
cd /Users/kevinloke/Desktop/Lottery\ Project/

# Run each Python script in sequence
python3 "Download Past Toto Result.py"
python3 "4DTotoLotteryV2.py"
python3 "4D LSTM Model.py"
python3 "SupremeTotoRandomForest.py"
python3 "4D Toto Prediction v4.py"

# Change directory to the location of the Python scripts for Magnum
cd /Users/kevinloke/Desktop/Lottery\ Project/Magnum/

# Run the Python script for Magnum
python3 "Magnum 4DTotoLotteryV2.py"
python3 "Magnum 4D Toto Prediction v4.py"
python3 "Magnum 4D LSTM Model.py"
python3 "Magnum LSTM Top 3 Prediction.py"


