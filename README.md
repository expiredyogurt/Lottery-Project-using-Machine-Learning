# Lottery-Random-Forest
This project attempts to predict lottery results using various machine learning models such as LSTM, XGBoost, Deep Learning, Neural Network, and Random Forest.

These models were run on three different lottery brands: Sports Toto, Magnum, and Da Ma Cai.

No past dataset is provided in this project; you can obtain it from public websites with a little crawling.

A shell script (.sh file) is used to run all the Python scripts sequentially. There is also a Python file to pull the winning numbers from the website and append them to the CSV file for future training of the models.

Data was trained on 20+ years of past lottery results, although it is not recommended to go that far back as the machines or drawing methods have changed over the years.

**DMC LSTM Model** - Uses the LSTM machine learning model to train, test, and predict.

**DMC Toto Prediction v4** - Uses the Random Forest and Neural Network machine learning models to train, test, and predict.

**DMC TotoLotteryV2** - Uses Deep Learning and XGBoost machine learning models to train, test, and predict.

**4D Lottery Checker V2** - Checks the respective websites for the winning numbers and cross-references them with the predicted dataset, highlighting the winning numbers and their permutations in yellow.
