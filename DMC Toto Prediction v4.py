#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


import pandas as pd
import random
from tqdm import tqdm  # Import tqdm for progress bar

# Set the seed value
random.seed(53846)

# Load the dataset
data = pd.read_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC Latest Result.csv')

# Remove leading and trailing whitespaces from column names
data.columns = data.columns.str.strip()

# Concatenate the values in columns 'DrawnNo1' to 'DrawnNo6' with commas
data['Concatenated'] = data[['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']].apply(lambda x: ','.join(map(str, x)), axis=1)

# Get unique sequences
unique_sequences = set(data['Concatenated'])

# Generate random sequences until 7649 unique ones are found
random_sequences = set()

# Use tqdm to create a progress bar
with tqdm(total=5000, desc="Generating Random Sequences") as pbar:
    while len(random_sequences) < 5000:
        random_sequence = ','.join(str(num) for num in random.sample(range(1, 10000), 23))
        if random_sequence not in unique_sequences:
            random_sequences.add(random_sequence)
            pbar.update(1)  # Update progress bar

# Create a DataFrame from the generated random sequences
random_data = pd.DataFrame([seq.split(',') for seq in random_sequences], columns=['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10'])

# Concatenate the values in columns 'DrawnNo1' to 'DrawnNo6' with commas for the newly generated data
random_data['Concatenated'] = random_data[['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']].apply(lambda x: ','.join(map(str, x)), axis=1)

# Concatenate the original data with the new random data
updated_data = pd.concat([data, random_data], ignore_index=True)

# Replace null values or empty strings in the 'Winnings' column with 'No' or empty whitespace
updated_data['Winnings'].fillna('No', inplace=True)
updated_data['Winnings'] = updated_data['Winnings'].str.strip()

# Save the updated data to a CSV file
updated_data.to_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC4D_Latest_Result_PreProcessed.csv', index=False)

# Display the shape of the updated DataFrame
print("Shape of the updated DataFrame:", updated_data.shape)


# In[3]:


# This code trains the model in Neutral Network


# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('/Users/kevinloke/Desktop/Lottery Project/DMC/DMC4D_Latest_Result_PreProcessed.csv')

# Replace empty cells or null values in the 'Winnings' column with 'No'
data['Winnings'].fillna('No', inplace=True)

# Map 'Yes' to 1 and 'No' to 0
data['Winnings'] = data['Winnings'].map({'Yes': 1, 'No': 0})

# Extract features (X) and target variable (y)
X = data[['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']].values  # Features
y = data['Winnings'].values  # Target variable

# Convert y to float32 (required by TensorFlow)
y = y.astype(np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53846)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model for binary classification
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Use sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with a progress bar
with tqdm(total=100, desc="Training Progress") as pbar:
    for epoch in range(100):
        model.fit(X_train_scaled, y_train, batch_size=32, verbose=0)
        pbar.update(1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)


# In[5]:


#this code is to generate the numbers for the prediction


# In[6]:


import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import pandas as pd
import os
from datetime import datetime  # Import datetime for getting today's date

# Set the seed value
random.seed(53846)

# Generate a large number of random sets of drawn numbers
num_sets = 1000000  # Number of sets to generate
random_drawn_numbers = np.random.randint(1, 10000, size=(num_sets, 23))  # Generate random numbers between 1 and 58 for 10,000 sets of 6 numbers

# Use tqdm to create a progress bar
with tqdm(total=num_sets, desc="Generating Predictions") as pbar:
    # Make predictions for the randomly generated sets of drawn numbers
    predicted_winnings = model.predict(random_drawn_numbers)
    pbar.update(num_sets)  # Update progress bar

# Combine the predicted winnings with the corresponding drawn numbers
selected_sets = list(zip(random_drawn_numbers, predicted_winnings))

# Sort the sets based on the predicted winnings in descending order
sorted_sets = sorted(selected_sets, key=lambda x: x[1], reverse=True)

# Take only the top 10 sets
top_10_sets = sorted_sets[:10]

# Convert the top 10 sets into a DataFrame
top_10_df = pd.DataFrame(top_10_sets, columns=['Drawn Numbers', 'Predicted Winnings'])

# Split 'Drawn Numbers' into separate columns
drawn_numbers_df = top_10_df['Drawn Numbers'].apply(pd.Series)

# Rename the columns
drawn_numbers_df.columns = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']

# Specify only the columns you want to keep
formatted_df = drawn_numbers_df[['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10']]


# Get today's date in the desired format
today_date = datetime.now().strftime('%d %b %Y')

# Define the file name with today's date as suffix
file_name = f'DMC NN Prediction {today_date}.csv'

# Specify the directory where you want to save the file
output_directory = '/Users/kevinloke/Desktop/Lottery Project/DMC/'

# Combine the directory and file name to create the full file path
output_file_path = os.path.join(output_directory, file_name)

# Save the formatted DataFrame to a CSV file
formatted_df.to_csv(output_file_path, index=False)


# In[ ]:





# In[7]:


# made modification so anything above this code is the real deal


# import pandas as pd
# import os
# from datetime import datetime
# 
# # Assuming 'selected_sets' contains the list of selected sets of numbers and their predicted winnings from Code 1
# 
# # Extract the top 10 sets with their numbers and predicted winnings
# num_set = sorted(selected_sets, key=lambda x: x[1], reverse=True)[:10]
# numbers = [set[0] for set in top_10_sets]
# predicted_winnings = [set[1] for set in top_10_sets]
# 
# # Convert the drawn numbers into a DataFrame with appropriate column names
# code1_output_df = pd.DataFrame(numbers, columns=['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6'])
# 
# # Add the 'Predicted Winnings' column to the DataFrame
# code1_output_df['Predicted Winnings'] = predicted_winnings
# 
# # Specify the directory where you want to save the file
# output_directory = 'S:/CHEQ/Lottery Prediction ML/'
# 
# # Get today's date in the desired format
# today_date = datetime.now().strftime('%d %b %Y')
# 
# # Define the file name with today's date as suffix
# file_name = f'Supreme Toto Prediction {today_date}.csv'
# 
# # Combine the directory and file name to create the full file path
# file_path = os.path.join(output_directory, file_name)
# 
# # Save the DataFrame with Code 1 output to a CSV file
# code1_output_df.to_csv(file_path, index=False)
# 

# import pandas as pd
# import os
# from datetime import datetime
# 
# # Assuming 'selected_sets' contains the list of selected sets of numbers and their predicted winnings from Code 1
# 
# # Extract the top 10 sets with their numbers and predicted winnings
# top_10_sets = sorted(selected_sets, key=lambda x: x[1], reverse=True)[:10]
# numbers = [set[0] for set in top_10_sets]
# 
# # Convert the drawn numbers into a DataFrame with appropriate column names
# code1_output_df = pd.DataFrame(numbers, columns=['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6'])
# 
# # Specify the directory where you want to save the file
# output_directory = 'S:/CHEQ/Lottery Prediction ML/'
# 
# # Get today's date in the desired format
# today_date = datetime.now().strftime('%d %b %Y')
# 
# # Define the file name with today's date as suffix
# file_name = f'Supreme Toto Prediction {today_date}.csv'
# 
# # Combine the directory and file name to create the full file path
# file_path = os.path.join(output_directory, file_name)
# 
# # Save the DataFrame with Code 1 output to a CSV file (excluding the 'Predicted Winnings' column)
# code1_output_df.to_csv(file_path, index=False, columns=['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6'])
# 

# In[8]:


#Use the code above to generate the final prediction. Anything below this line is for reference / play play only 


# import numpy as np
# import pandas as pd
# import random
# from tqdm import tqdm
# 
# # Set the seed value
# random.seed(53846)
# 
# # Define a threshold for predicted winnings
# threshold = 90
# 
# # Counter for the number of sets with predicted winnings above the threshold
# num_sets_found = 0
# 
# # Lists to store the sets of numbers and their corresponding predicted winnings
# selected_sets = []
# 
# # Define the total number of iterations needed to find 10 sets above the threshold
# total_iterations = 100
# 
# # Use tqdm to create a progress bar
# with tqdm(total=total_iterations, desc="Generating Sets") as pbar:
#     # Loop until 10 sets are found above the threshold or the total iterations are reached
#     while num_sets_found < 10 and pbar.n < total_iterations:
#         # Generate a random set of drawn numbers without repetition
#         random_drawn_numbers = random.sample(range(1, 59), 6)
# 
#         # Convert the random set of drawn numbers to a DataFrame with appropriate column names
#         random_drawn_numbers_df = pd.DataFrame([random_drawn_numbers], columns=['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6'])
# 
#         # Make a prediction for the random set of drawn numbers
#         predicted_winnings = model.predict(random_drawn_numbers_df)
# 
#         # Check if predicted winnings are above the threshold
#         if predicted_winnings > threshold:
#             # Increment the counter for sets found above the threshold
#             num_sets_found += 1
# 
#             # Store the set of drawn numbers and the predicted winnings
#             selected_sets.append((random_drawn_numbers, predicted_winnings[0]))
# 
#         # Update the progress bar
#         pbar.update(1)
# 
# # Print the selected sets of numbers and their predicted winnings
# print("Selected Sets:")
# for i, (numbers, winnings) in enumerate(selected_sets, start=1):
#     print(f"Set {i}: Drawn Numbers: {numbers}, Predicted Winnings: {winnings}")
# 

# In[9]:


# This code uses Random Forest Regression


# import numpy as np
# import pandas as pd
# import random
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from tqdm import tqdm
# from os import cpu_count
# from datetime import datetime  # Import datetime for getting today's date
# 
# # Set the seed value
# random.seed(53846)
# 
# import warnings
# from sklearn.exceptions import DataConversionWarning
# 
# # Filter out the warning regarding feature names
# warnings.filterwarnings("ignore", category=UserWarning)
# 
# 
# # Define a threshold for predicted winnings
# threshold = 0.95
# 
# # Function to generate random sets of numbers and make predictions
# def generate_and_predict(_):
#     # Generate a random set of drawn numbers without repetition
#     random_drawn_numbers = random.sample(range(1, 10000), 23)
# 
#     # Convert the random set of drawn numbers to a DataFrame with appropriate column names
#     random_drawn_numbers_df = pd.DataFrame([random_drawn_numbers], columns=['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10'])
# 
#     # Make a prediction for the random set of drawn numbers
#     predicted_winnings = model.predict(random_drawn_numbers_df)
#     
#     # Check if predicted winnings exceed the threshold
#     if predicted_winnings[0] >= threshold:
#         return random_drawn_numbers, predicted_winnings[0]
#     else:
#         return None, None
# 
# 
# # Number of sets to generate
# num_sets = 100000
# 
# # Use tqdm to create a progress bar
# with tqdm(total=num_sets, desc="Generating and Predicting") as pbar:
#     # Create a Pipeline with the model
#     pipe = Pipeline([('model', RandomForestRegressor())])
# 
#     # Define the parameter grid for hyperparameter tuning
#     param_grid = {
#         'model__n_estimators': [100, 200, 300],
#         'model__max_depth': [None, 10, 20]
#     }
# 
#     # Create the GridSearchCV object with Parallel execution
#     grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
# 
#     # Fit the GridSearchCV object
#     grid_search.fit(X_train, y_train)
# 
#     # Get the best model from the GridSearchCV
#     model = grid_search.best_estimator_
# 
#     # Generate and predict random sets of numbers
#     selected_sets = []
#     for _ in range(num_sets):
#         numbers, winnings = generate_and_predict(_)
#         if numbers is not None:
#             selected_sets.append((numbers, winnings))
#             pbar.update(1)  # Update progress bar
#             
# # Sort the selected sets based on predicted winnings in descending order
# selected_sets.sort(key=lambda x: x[1], reverse=True)
# 
# # Select the top 10 sets
# top_10_sets = selected_sets[:10]
# 
# # Convert the top 10 sets to a DataFrame
# df_top_10 = pd.DataFrame(top_10_sets, columns=['Drawn Numbers', 'Predicted Winnings'])
# 
# # Convert the 'Drawn Numbers' column to a DataFrame of drawn numbers with specific column labels
# drawn_numbers_df = pd.DataFrame(df_top_10['Drawn Numbers'].tolist(), columns=['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10'])
# 
# # Concatenate the drawn numbers DataFrame with the existing DataFrame
# df_top_10 = pd.concat([df_top_10, drawn_numbers_df], axis=1)
# 
# # Drop the 'Drawn Numbers' column
# df_top_10.drop(columns=['Drawn Numbers'], inplace=True)
# 
# 
# # Save the DataFrame to a CSV file
# #df_top_10.to_csv('S:/CHEQ/Lottery Prediction ML/top_10_predicted_winnings.csv', index=False)
# 
# # Get today's date
# today_date = datetime.now().strftime('%d %b %Y')
# 
# # Define the filename with today's date
# filename = f'/Users/kevinloke/Desktop/Lottery Project/4D Random Forest Reg Prediction {today_date}.csv'
# 
# # Save the DataFrame to a CSV file
# df_top_10.to_csv(filename, index=False)
# 
# # Print the selected sets of numbers and their predicted winnings
# print("Top 10 Sets:")
# for i, (numbers, winnings) in enumerate(top_10_sets, start=1):
#     print(f"Set {i}: Drawn Numbers: {numbers}, Predicted Winnings: {winnings}")
# 

# In[10]:


# this uses MODIN to speed things up, still testing it


# In[11]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from datetime import datetime

# Set the seed value
np.random.seed(53846)

# Define a threshold for predicted winnings
threshold = 0.5

# Number of sets to generate
num_sets = 1000000

# Generate random sets of numbers
random_drawn_numbers = np.random.randint(1, 10000, size=(num_sets, 23))

# Create a Pipeline with the model
pipe = Pipeline([('model', RandomForestRegressor())])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20]
}

# Create the GridSearchCV object with Parallel execution
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train)

# Get the best model from the GridSearchCV
model = grid_search.best_estimator_

# Initialize tqdm for progress bar
with tqdm(total=num_sets, desc="Generating and Predicting") as pbar:
    # Initialize an empty list to store selected sets
    selected_sets = []

    # Batch prediction to improve performance
    batch_size = 1000
    num_batches = num_sets // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_sets)
        batch_numbers = random_drawn_numbers[start_idx:end_idx]

        # Make predictions for the batch of random sets
        predicted_winnings = model.predict(batch_numbers)

        # Check if predicted winnings exceed the threshold
        for j, winnings in enumerate(predicted_winnings):
            if winnings >= threshold:
                selected_sets.append((batch_numbers[j], winnings))
            pbar.update(1)  # Update progress bar

# Sort the selected sets based on predicted winnings in descending order
selected_sets.sort(key=lambda x: x[1], reverse=True)

# Select the top 10 sets
top_10_sets = selected_sets[:10]

# Convert the top 10 sets to a DataFrame
df_top_10 = pd.DataFrame(top_10_sets, columns=['Drawn Numbers', 'Predicted Winnings'])

# Convert the 'Drawn Numbers' column to a DataFrame of drawn numbers with specific column labels
drawn_numbers_df = pd.DataFrame(df_top_10['Drawn Numbers'].tolist(), columns=[
    '1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo', 
    'SpecialNo1', 'SpecialNo2', 'SpecialNo3', 'SpecialNo4', 'SpecialNo5', 'SpecialNo6', 'SpecialNo7', 'SpecialNo8', 'SpecialNo9', 'SpecialNo10', 
    'ConsolationNo1', 'ConsolationNo2', 'ConsolationNo3', 'ConsolationNo4', 'ConsolationNo5', 'ConsolationNo6', 'ConsolationNo7', 'ConsolationNo8', 'ConsolationNo9', 'ConsolationNo10'])

# Concatenate the drawn numbers DataFrame with the existing DataFrame
df_top_10 = pd.concat([df_top_10, drawn_numbers_df], axis=1)

# Drop the 'Drawn Numbers' column
df_top_10.drop(columns=['Drawn Numbers'], inplace=True)


# Get today's date
today_date = datetime.now().strftime('%d %b %Y')

# Define the filename with today's date
filename = f'/Users/kevinloke/Desktop/Lottery Project/DMC/DMC Random Forest Reg Prediction {today_date}.csv'

# Save the DataFrame to a CSV file
df_top_10.to_csv(filename, index=False)

# Print the selected sets of numbers and their predicted winnings
print("Top 10 Sets:")
for i, (numbers, winnings) in enumerate(top_10_sets, start=1):
    print(f"Set {i}: Drawn Numbers: {numbers}, Predicted Winnings: {winnings}")


# In[12]:


#Save the top 10 into a CSV file 


# # Convert the top_10_sets to a DataFrame
# df_top_10 = pd.DataFrame([numbers for numbers, _ in top_10_sets], columns=['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6'])
# 
# 
# # Get today's date
# today_date = datetime.now().strftime('%d %b %Y')
# 
# # Define the filename with today's date
# filename = f'S:/CHEQ/Lottery Prediction ML/Supreme Random Forest Reg Prediction {today_date}.csv'
# 
# # Save the DataFrame to a CSV file
# df_top_10.to_csv(filename, index=False)
# 

# In[13]:


#this code extract the winning numbers from the website to test if its working


# In[14]:






