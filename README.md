import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np

# Step 1: Define car colors and create a database

# Predefined list of car colors and their popularity (in percentage)
car_colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Green', 'Yellow', 'Gray', 'Purple', 'Pink']
color_data = {
    'Color': car_colors,
    'Popularity': [12, 8, 25, 20, 15, 5, 2, 10, 2, 1]  # Example: percentage popularity of each color
}

# Create a pandas DataFrame
df = pd.DataFrame(color_data)

# Display the color popularity
print("Car Color Popularity:")
print(df)

# Step 2: Simulate car data with car colors, car types, and user preferences

car_types = ['Sedan', 'SUV', 'Truck', 'Coupe', 'Hatchback']
users = ['Alice', 'Bob', 'Charlie', 'David', 'Emma']

# Randomly generate car data
car_data = []
for i in range(1000):
    car_type = random.choice(car_types)
    car_color = random.choice(car_colors)
    user = random.choice(users)
    car_data.append([user, car_type, car_color])

# Convert to DataFrame
car_df = pd.DataFrame(car_data, columns=['User', 'CarType', 'Color'])
print("\nSample Car Data:")
print(car_df.head())

# Step 3: Analyze color preferences based on car type and user demographics

# Grouping by car type and color to see the distribution of colors
color_distribution_by_type = car_df.groupby(['CarType', 'Color']).size().unstack().fillna(0)

# Display the distribution
print("\nCar Color Distribution by Car Type:")
print(color_distribution_by_type)

# Step 4: Visualize car color popularity with a pie chart
plt.figure(figsize=(7, 7))
plt.pie(df['Popularity'], labels=df['Color'], autopct='%1.1f%%', startangle=90)
plt.title('Car Color Popularity')
plt.show()

# Step 5: Machine Learning - Predict car color preference based on car type and user demographics

# Encode categorical data (CarType, Color, User)
label_encoder = LabelEncoder()
car_df['CarType_encoded'] = label_encoder.fit_transform(car_df['CarType'])
car_df['Color_encoded'] = label_encoder.fit_transform(car_df['Color'])
car_df['User_encoded'] = label_encoder.fit_transform(car_df['User'])

# Features and target
X = car_df[['CarType_encoded', 'User_encoded']]
y = car_df['Color_encoded']

# Train a simple logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
sample_input = pd.DataFrame([[label_encoder.transform(['SUV'])[0], label_encoder.transform(['Alice'])[0]]], columns=['CarType_encoded', 'User_encoded'])
predicted_color_encoded = model.predict(sample_input)
predicted_color = label_encoder.inverse_transform(predicted_color_encoded)

print("\nPredicted car color for an 'SUV' chosen by 'Alice':", predicted_color[0])

# Step 7: Visualize the machine learning results using a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict on the test set
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Car Color Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 8: Track trends in car color preferences over time (simulated)

# Simulate a trend over years
years = list(range(2010, 2025))
trend_data = {color: [random.randint(50, 500) for _ in years] for color in car_colors}

# Create a DataFrame for trends
trend_df = pd.DataFrame(trend_data, index=years)

# Plot the trends
trend_df.plot(figsize=(12, 8))
plt.title('Car Color Trends Over Time')
plt.ylabel('Number of Cars')
plt.xlabel('Year')
plt.legend(title='Car Colors')
plt.grid(True)
plt.show()

# Step 9: Car color suggestions based on car type
def suggest_color_based_on_car_type(car_type):
    if car_type == 'Sedan':
        return random.choice(['Black', 'White', 'Silver'])
    elif car_type == 'SUV':
        return random.choice(['Blue', 'Black', 'Gray'])
    elif car_type == 'Truck':
        return random.choice(['Red', 'Black', 'Silver'])
    elif car_type == 'Coupe':
        return random.choice(['Red', 'White', 'Black'])
    else:
        return random.choice(['Green', 'Yellow', 'Silver'])

# Test the suggestion function
car_type = 'SUV'
suggested_color = suggest_color_based_on_car_type(car_type)
print(f"\nSuggested car color for a {car_type}: {suggested_color}")

# Step 10: Enhancing the User Interface for Interactivity

def get_user_input():
    print("\nWelcome to the Car Color Management System!")
    user_name = input("Enter your name: ")
    car_type = input("Enter car type (Sedan, SUV, Truck, Coupe, Hatchback): ")
    
    if car_type not in car_types:
        print(f"Invalid car type. Defaulting to 'Sedan'.")
        car_type = 'Sedan'
        
    print("\nFetching car color preferences...")
    suggested_color = suggest_color_based_on_car_type(car_type)
    print(f"\n{user_name}, based on the car type '{car_type}', we suggest the color: {suggested_color}.\n")
    
    return user_name, car_type, suggested_color

# Step 11: Simulate users interacting with the system
user_interactions = []
for _ in range(5):
    user_data = get_user_input()
    user_interactions.append(user_data)

# Step 12: Add more complexity - user demographics analysis
user_demographics = pd.DataFrame(user_interactions, columns=['UserName', 'CarType', 'SuggestedColor'])
print("\nUser Demographics Analysis:")
print(user_demographics)

# Step 13: Analyzing the correlation between car types and the suggested colors
car_color_analysis = user_demographics.groupby(['CarType', 'SuggestedColor']).size().unstack().fillna(0)
print("\nCar Color Analysis by Car Type:")
print(car_color_analysis)

# Step 14: Track the popularity of car colors based on car type over time
time_series_data = {year: {car_type: random.randint(10, 50) for car_type in car_types} for year in range(2010, 2025)}
time_series_df = pd.DataFrame(time_series_data)

# Step 15: Plot the trend over years for car types
time_series_df.plot(figsize=(12, 8))
plt.title('Car Type Popularity Over Time')
plt.ylabel('Number of Cars Sold')
plt.xlabel('Year')
plt.legend(title='Car Types')
plt.grid(True)
plt.show()

# Step 16: Random Forest Regression for Predicting Car Color Preferences
from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Confusion matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_pred)
rf_cm_df = pd.DataFrame(rf_cm, index=label_encoder.classes_, columns=label_encoder.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(rf_cm_df, annot=True, fmt='d', cmap='Purples')
plt.title('Random Forest Confusion Matrix for Car Color Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 17: Save Data for Future Use
df.to_csv('car_colors_data.csv', index=False)
car_df.to_csv('car_data.csv', index=False)

print("\nData has been saved successfully.")

# Final Notes:
# This script performs a variety of functions related to car color management.
# It includes data analysis, predictive modeling, and visualizations, making it a comprehensive
# system for understanding car color preferences and trends over time.
