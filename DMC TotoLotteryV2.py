#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ________                         .____                              .__                
# \______ \   ____   ____ ______   |    |    ____ _____ _______  ____ |__| ____    ____  
#  |    |  \_/ __ \_/ __ \\____ \  |    |  _/ __ \\__  \\_  __ \/    \|  |/    \  / ___\ 
#  |    `   \  ___/\  ___/|  |_> > |    |__\  ___/ / __ \|  | \/   |  \  |   |  \/ /_/  >
# /_______  /\___  >\___  >   __/  |_______ \___  >____  /__|  |___|  /__|___|  /\___  / 
#         \/     \/     \/|__|             \/   \/     \/           \/        \//_____/  


# In[2]:


#Run this first to preprocess the txt file into csv


# In[3]:


# In[4]:


#this is the main code to fill the dataset with random numbers in the correct format with a progress bar 


# In[5]:


import random
import csv
from tqdm import tqdm
import numpy as np 

# Set the seed value
np.random.seed(53846)
# Set the random seed for reproducibility
random.seed(53846)  # You can change the seed value as needed

def read_existing_numbers(file_path):
    existing_numbers = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            existing_numbers.append(row)
    return existing_numbers

def find_first_empty_cell(existing_numbers):
    for idx, row in enumerate(existing_numbers):
        if not row[0]:
            return idx
    return len(existing_numbers)  # If no empty cell found, append at the end

def generate_lottery_numbers(existing_numbers, start_idx, total):
    new_numbers = []
    with tqdm(total=total) as pbar:
        while len(new_numbers) < total:
            new_row = []
            for _ in range(23):
                num = str(random.randint(1, 9999)).zfill(4)
                while num in [item for sublist in existing_numbers for item in sublist]:
                    num = str(random.randint(1, 9999)).zfill(4)
                new_row.append(num)
            new_numbers.append([''] * 2 + new_row)  # Move two columns to the right
            pbar.update(1)
    return existing_numbers[:start_idx] + new_numbers + existing_numbers[start_idx:]

def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

existing_numbers = read_existing_numbers('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC Latest Result.csv')
empty_cell_index = find_first_empty_cell(existing_numbers)
total_new_numbers = 5000
new_combined_data = generate_lottery_numbers(existing_numbers, empty_cell_index, total_new_numbers)

write_to_csv(new_combined_data, '/Users/kevinloke/Desktop/Lottery Project/DMC/DMC_updated_data_for_train_test.csv')


# In[6]:


#this code trims the whitespace and replace all null values in the winnings column into NO


# In[7]:


import pandas as pd

# Load the new dataset
new_data = pd.read_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC_updated_data_for_train_test.csv')

# Trim leading and trailing whitespaces from column names
new_data.columns = new_data.columns.str.strip()

# Fill missing values in the 'Winnings' column with 'No'
new_data['Winnings'].fillna('No', inplace=True)

# Save the modified DataFrame back to the same CSV file
new_data.to_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC_updated_data_for_train_test.csv', index=False)

# View the first few rows of the DataFrame
print("Head of the DataFrame:")
print(new_data.head())

# View the last few rows of the DataFrame
print("\nTail of the DataFrame:")
print(new_data.tail())


# In[8]:


#Train test the deep learning model


# In[9]:



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC_updated_data_for_train_test.csv')

# Replace 'Yes' with 1 and 'No' with 0 in the 'Winnings' column
data['Winnings'] = data['Winnings'].map({'Yes': 1, 'No': 0})

# Replace empty cells with NaN
data['Winnings'].replace('', float('nan'), inplace=True)

# Replace null values in the 'Winnings' column with 0
data['Winnings'].fillna(0, inplace=True)



# Extract features (X) and target variable (y)
X = data[['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']].values  # Features
y = data['Winnings'].values  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53846)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with 1 unit for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_classes = (y_pred > 0.95).astype(int)  # Convert probabilities to binary predictions
accuracy = model.evaluate(X_test_scaled, y_test)[1]

print("Accuracy:", accuracy)


# In[10]:


#this code is to run a test sample generating 1 set of number to see if it is working fine


# In[11]:


import numpy as np

np.random.seed(53846)

# Generate a random input sample (you can replace this with your own data)
# For example, you can use the mean or median values of each feature from the training dataset
random_input = np.random.randn(1, X_train_scaled.shape[1])  # Assuming X_train_scaled is standardized

# Predict the probability of winning for the random input
winning_probability = model.predict(random_input)[0][0]

# Define a threshold (you can adjust this threshold based on your preference)
threshold = 0.95

# Generate 23 numbers based on the winning probability
if winning_probability >= threshold:
    # Generate winning numbers
    generated_numbers = np.random.randint(1, 10000, size=23)
else:
    # Generate non-winning numbers
    generated_numbers = np.zeros(23)  # You can adjust this logic based on your requirements

print("Generated Numbers:", generated_numbers)


# In[12]:


#This is the final code to run to generate 10 sets of numbers with proper table format and meets the threshold 


# In[13]:


import numpy as np
import pandas as pd

np.random.seed(53846)

# Define the number of sets to generate
num_sets = 10

# Initialize an empty list to store the generated numbers
all_generated_numbers = []

# Continue generating numbers until there are 10 sets that pass the threshold
while len(all_generated_numbers) < num_sets:
    # Generate a random input sample (you can replace this with your own data)
    # For example, you can use the mean or median values of each feature from the training dataset
    random_input = np.random.randn(1, X_train_scaled.shape[1])  # Assuming X_train_scaled is standardized

    # Predict the probability of winning for the random input
    winning_probability = model.predict(random_input)[0][0]

    # Define a threshold (you can adjust this threshold based on your preference)
    threshold = 0.95

    # Check if the winning probability exceeds the threshold
    if winning_probability >= threshold:
        # Generate 23 numbers based on the winning probability
        generated_numbers = np.random.randint(1, 10000, size=23)
        
        # Append the generated numbers to the list
        all_generated_numbers.append(generated_numbers)

# Create a DataFrame from the list of generated numbers
columns = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']
generated_numbers_df = pd.DataFrame(all_generated_numbers, columns=columns)

# Print the DataFrame
print(generated_numbers_df)


# In[14]:


#this code save the output of DeepL into a CSV with proper formatting


# In[15]:


import os  # Import the os module
from datetime import datetime

# Get today's date in the desired format
today_date = datetime.now().strftime('%d %b %Y')

# Define the file name with today's date as suffix
file_name = f'DMC DeepL Prediction {today_date}.csv'

# Specify the full path including the directory where you want to save the file
output_directory = '/Users/kevinloke/Desktop/Lottery Project/DMC/'
file_path = os.path.join(output_directory, file_name)

# Save the predictions to a CSV file
generated_numbers_df.to_csv(file_path, index=False)

print("Predictions saved to:", file_path)


# In[16]:


# ____  _____________________                       __   
# \   \/  /  _____/\______   \ ____   ____  _______/  |_ 
#  \     /   \  ___ |    |  _//  _ \ /  _ \/  ___/\   __\
#  /     \    \_\  \|    |   (  <_> |  <_> )___ \  |  |  
# /___/\  \______  /|______  /\____/ \____/____  > |__|  
#       \_/      \/        \/                  \/        


# In[17]:


#This code train test the model


# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Set the seed value
np.random.seed(53846)

# Load the dataset
data = pd.read_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC_updated_data_for_train_test.csv')

# Replace 'Yes' with 1 and 'No' with 0 in the 'Winnings' column
data['Winnings'] = data['Winnings'].map({'Yes': 1, 'No': 0})

# Replace null values in the 'Winnings' column with 0
data['Winnings'].fillna(0, inplace=True)

# Extract features (X) and target variable (y)
X = data[['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']].values  # Features
y = data['Winnings'].values  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53846)

# Standardize the features
XGB_scaler = StandardScaler()  # Changed variable name from scaler to XGB_scaler
X_train_scaled = XGB_scaler.fit_transform(X_train)  # Changed variable name from scaler to XGB_scaler
X_test_scaled = XGB_scaler.transform(X_test)  # Changed variable name from scaler to XGB_scaler

# Define the XGBoost model with the seed number
XGB_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Train the XGBoost model
XGB_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_XGB = XGB_model.predict(X_test_scaled)

# Calculate accuracy
accuracy_XGB = accuracy_score(y_test, y_pred_XGB)

print("XGBoost Accuracy:", accuracy_XGB)


# In[19]:


#this code train test the model and make the prediction


# In[20]:


import pandas as pd
import random
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Set the seed value
np.random.seed(53846)
# Set the random seed for reproducibility
random.seed(53846)  # You can change the seed value as needed

# Initialize an empty list to store the top sets of numbers
XGB_top_sets = []

# Assuming `XGB_model` is your trained XGBoost model and `XGB_scaler` is your StandardScaler object
while len(XGB_top_sets) < 10:  # Number of sets to generate
    # Generate a set of lottery numbers
    generated_numbers = np.random.randint(1, 10000, size=(1, 23))  # Generate 23 numbers
    
    # Standardize the generated numbers using the same scaler used for training the XGBoost model
    generated_numbers_scaled = XGB_scaler.transform(generated_numbers)
    
    # Make predictions for the generated numbers using the trained XGBoost model
    predicted_winnings = XGB_model.predict(generated_numbers_scaled)
    
    # Compute the confidence level based on the maximum prediction value
    confidence_level = np.max(predicted_winnings)
    
    # If the confidence level is 95% or higher, add the generated numbers and predictions to the list
    if confidence_level >= 0.9999:
        XGB_top_sets.append((generated_numbers.flatten(), confidence_level))

# Create a DataFrame to display the top sets of numbers and their corresponding confidence levels
XGB_top_sets_df = pd.DataFrame(XGB_top_sets, columns=['Lottery Numbers', 'Confidence Level'])

# Sort the DataFrame based on the confidence level in descending order
XGB_top_sets_df = XGB_top_sets_df.sort_values(by='Confidence Level', ascending=False).reset_index(drop=True)

# Print the top sets of numbers and their confidence levels
print(XGB_top_sets_df)


# In[21]:


#This code saves it to CSV with the current date


# In[22]:


import pandas as pd
import os
from datetime import datetime

# Assuming XGB_top_sets_df contains the DataFrame with lottery numbers and confidence levels

# Extract the 'Lottery Numbers' column from the DataFrame
XGB_lottery_numbers_df = pd.DataFrame(XGB_top_sets_df['Lottery Numbers'].to_list(), columns=[
    '1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 
    'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 
    'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10'
])

# Specify the directory where you want to save the file
output_directory = '/Users/kevinloke/Desktop/Lottery Project/DMC/'

# Get today's date in the desired format
today_date = datetime.now().strftime('%d %b %Y')

# Define the file name with today's date as suffix
file_name = f'DMC XGB Prediction {today_date}.csv'

# Combine the directory and file name to create the full file path
file_path = os.path.join(output_directory, file_name)

# Save the DataFrame with lottery numbers to a CSV file
XGB_lottery_numbers_df.to_csv(file_path, index=False)


# In[ ]:

