#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Limit the dataset to 1 year ago from today


# In[2]:


import pandas as pd
from datetime import datetime, timedelta

# Read the CSV file
df = pd.read_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC Latest Result.csv')

# Loop through each column in the DataFrame
for column in df.columns:
    # Check if the data type of the column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        # Format numbers to have four digits
        df[column] = df[column].apply(lambda x: '{:04d}'.format(int(x)) if not pd.isnull(x) else x)

# Convert 'DrawDate' column to datetime object
df['DrawDate'] = pd.to_datetime(df['DrawDate'], format='%Y-%m-%d')

# Calculate the threshold date (today's date - 12 months)
threshold_date = datetime.now() - timedelta(days=99999)

# Filter out rows where 'DrawDate' is greater than threshold date
df = df[df['DrawDate'] > threshold_date]

# Save the modified DataFrame back to a CSV file named '4DPP.csv'
df.to_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC4DPP.csv', index=False)

df.head()

df.tail()


# In[3]:


#filtered the dates and save it to a file called 4DPP


# In[4]:


import pandas as pd

# Assuming your original dataset is stored in 'original_dataset.csv'
# Load the original dataset
original_dataset = pd.read_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC4DPP.csv')

# Select only the desired columns
selected_columns = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 
                    'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 
                    'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 
                    'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 
                    'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 
                    'ConsolationNo10']

selected_data = original_dataset[selected_columns]

# Save the selected data to a new CSV file named '4DPP.csv'
selected_data.to_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC4DPP.csv', index=False)

selected_data.tail()


# In[5]:


# In[11]:


#bump it up to 10x numbers generate


# In[12]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art
from datetime import datetime

today_date = datetime.now().strftime('%d %b %Y')
filename = f"/Users/kevinloke/Desktop/Lottery Project/DMC/DMC LotteryAI Prediction {today_date}.csv"



# Set seed for reproducibility
seed_value = 53846
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Function to print the introduction of the program
def print_intro():
    # Generate ASCII art with the text "LotteryAi"
    ascii_art = text2art("LotteryAi")
    # Print the introduction and ASCII art
    print("============================================================")
    print("============================================================")
    print(ascii_art)
    print("Lottery prediction artificial intelligence")

# Function to load data from a file and preprocess it
def load_data():
    # Load data from file, ignoring white spaces and accepting unlimited length numbers
    data = np.genfromtxt('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC4DPP.csv', delimiter=',', dtype=int)
    # Replace all -1 values with 0
    data[data == -1] = 0
    # Split data into training and validation sets
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):]
    # Get the maximum value in the data
    max_value = np.max(data)
    return train_data, val_data, max_value

# Function to create the model
def create_model(num_features, max_value):
    # Create a sequential model
    model = keras.Sequential()
    # Add an Embedding layer, LSTM layer, and Dense layer to the model
    model.add(layers.Embedding(input_dim=max_value+1, output_dim=64))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(num_features, activation='softmax'))
    # Compile the model with categorical crossentropy loss, adam optimizer, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, train_data, val_data):
    # Fit the model on the training data and validate on the validation data for 100 epochs
    history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100)

# Function to predict numbers using the trained model
def predict_numbers(model, val_data, num_features, num_sets=10):
    all_predicted_numbers = []
    for _ in range(num_sets):
        # Predict on the validation data using the model
        predictions = model.predict(val_data)
        # Get the indices of the top 'num_features' predictions for each sample in validation data
        indices = np.argsort(predictions, axis=1)[:, -num_features:]
        # Get the predicted numbers using these indices from validation data
        predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
        # Take only the first row of predicted numbers to form a set
        all_predicted_numbers.append(predicted_numbers[0])
        # Shuffle the validation data for the next iteration
        np.random.shuffle(val_data)
    return np.array(all_predicted_numbers)



# Function to save predicted numbers to a CSV file with today's date in the filename
def save_predicted_numbers_to_csv(all_predicted_numbers, header_labels):
    today_date = datetime.now().strftime('%d %b %Y')
    filename = f"/Users/kevinloke/Desktop/Lottery Project/DMC/DMC LotteryAI Prediction {today_date}.csv"
    
    # Add the header labels as a separate row in the CSV file
    with open(filename, 'w') as file:
        file.write(','.join(header_labels) + '\n')
        # Save the predicted numbers to the CSV file
        np.savetxt(file, all_predicted_numbers, delimiter=",", fmt='%d')
    
    print(f"Predicted numbers saved to {filename}")




# Function to print the predicted numbers
def print_predicted_numbers(all_predicted_numbers, num_sets=10):
   # Print a separator line and "Predicted Numbers:"
   print("============================================================")
   print("Predicted Numbers for each set:")
   for set_num in range(num_sets):
       print(f"Set {set_num + 1}: {', '.join(map(str, all_predicted_numbers[set_num]))}")
   print("============================================================")


# Main function to run everything   
def main():
   # Print introduction of program 
   print_intro()
   
   # Load and preprocess data 
   train_data, val_data, max_value = load_data()
   
   # Get number of features from training data 
   num_features = train_data.shape[1]
   
   # Create and compile model 
   model = create_model(num_features, max_value)
   
   # Train model 
   train_model(model, train_data, val_data)
   
    # Predict numbers using trained model 
   all_predicted_numbers = predict_numbers(model, val_data, num_features)

   # Print predicted numbers 
   print_predicted_numbers(all_predicted_numbers)

   # Call the function with the desired file name
   today_date = datetime.now().strftime('%d %b %Y')
   filename = f"/Users/kevinloke/Desktop/Lottery Project/DMC/DMC LotteryAI Prediction {today_date}.csv"
   
    # Header labels for the CSV file
   header_labels = [
        '1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4',
        'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1',
        'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7',
        'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10'
    ]
   
 # Save predicted numbers to CSV
   save_predicted_numbers_to_csv(all_predicted_numbers, header_labels)

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
   main()


# In[13]:


