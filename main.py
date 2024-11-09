import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
crime_data_path = './crime_data.csv'
crime_clusters_path = './crime_clusters.csv'

# Load the crime_data CSV
crime_data_df = pd.read_csv(crime_data_path, low_memory=False)
crime_data_df["total_ipc_crimes"] = crime_data_df["total_ipc_crimes"].fillna(0)

# Standardize column names for consistency across both datasets
crime_data_df.columns = [col.lower() for col in crime_data_df.columns]

# Group the data by state and year, and by district and year
grouped_state_year = crime_data_df.groupby(['state_ut', 'year']).sum().reset_index()
grouped_district_year = crime_data_df.groupby(['district', 'year']).sum().reset_index()

# Load the crime_clusters CSV
crime_clusters_df = pd.read_csv(crime_clusters_path, encoding='ISO-8859-1', low_memory=False)
crime_clusters_df.columns = [col.lower() for col in crime_clusters_df.columns]

# Group the data by clusters, state, and year
clustered_group = crime_clusters_df.groupby(['state', 'year']).sum().reset_index()

# Create a dictionary of states with yearly crime counts
state_crime_totals = grouped_state_year.groupby('state_ut')['total_ipc_crimes'].apply(list).to_dict()

# Helper functions to retrieve data based on input
def get_by_state_year(state, year):
    result = grouped_state_year[(grouped_state_year['state_ut'] == state) & (grouped_state_year['year'] == year)]
    return result['total_ipc_crimes'].values[0] if not result.empty else 0

def get_by_district_crimes(district, crimes):
    return grouped_district_year.loc[grouped_district_year['district'] == district, crimes].values[0]

def get_by_district_year(district, year):
    result = grouped_district_year[(grouped_district_year['district'] == district) & (grouped_district_year['year'] == year)]
    return result['total_ipc_crimes'].values[0] if not result.empty else 0

def get_by_cluster_state_year(cluster, state, year):
    result = clustered_group[(clustered_group['cluster'] == cluster) & (clustered_group['state'] == state) & (clustered_group['year'] == year)]
    return result.to_dict('records') if not result.empty else {}

def get_by_state(state):
    return state_crime_totals.get(state, [])

# Enhanced Helper function for chatbot
def helper(entities):
    required = []
    for key in entities:
        if entities[key]:
            required.append(key)

    if not required:
        return 'Please input the query using State, District, Year, or Crime Type.'

    # Check for different conditions based on input entities
    if 'geo-state' in required and 'date-period' in required and 'crimes' in required:
        val = get_by_state_year(entities['geo-state'].lower(), entities['date-period'])
        return f"{entities['crimes']} - {val}. Results for provided inputs."
    
    elif 'geo-state' in required and 'date-period' in required:
        val = get_by_state_year(entities['geo-state'].lower(), entities['date-period'])
        return f"{val} crimes in {entities['geo-state']} for year {entities['date-period']}."
    
    elif 'geo-city' in required and 'crimes' in required and 'date-period' in required:
        val = get_by_district_year(entities['geo-city'].lower(), entities['date-period'])
        return f"{val} crimes in {entities['geo-city']} for year {entities['date-period']}."
    
    elif 'geo-cluster' in required and 'geo-state' in required and 'date-period' in required:
        val = get_by_cluster_state_year(entities['geo-cluster'], entities['geo-state'], entities['date-period'])
        return f"{val} crimes in cluster {entities['geo-cluster']} for {entities['geo-state']} in {entities['date-period']}."
    
    return "Please provide valid information. You may include parameters like State, District, Year, or Crime Type for a more accurate query."

# Label encoding for 'attack type' as target variable with binary labels
crime_clusters_df['deceptive'] = crime_clusters_df['attack type'].apply(lambda x: 1 if 'suspected' in x.lower() else 0)

# Select relevant features and encode categorical variables
crime_clusters_df = pd.get_dummies(crime_clusters_df, columns=['state', 'city'], drop_first=True)

# Train-test split
X = crime_clusters_df.drop(['deceptive', 'attack type'], axis=1)
y = crime_clusters_df['deceptive']
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Logistic Regression Model
# lr_model = LogisticRegression(random_state=42, max_iter=1000)
# lr_model.fit(X_train, y_train)
# y_pred_lr = lr_model.predict(X_test)

# print("\nLogistic Regression Model")
# print("Accuracy:", accuracy_score(y_test, y_pred_lr))
# print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# To further compare models
# def evaluate_models():
#     rf_accuracy = accuracy_score(y_test, y_pred_rf)
#     lr_accuracy = accuracy_score(y_test, y_pred_lr)
#     return f"Random Forest Accuracy: {rf_accuracy:.2f}, Logistic Regression Accuracy: {lr_accuracy:.2f}"

# Example usage of chatbot helper function
# entities = { 'geo-state': 'Maharashtra', 'date-period': 2015, 'crimes': 'Murder' }
# print(helper(entities))




from sklearn.model_selection import train_test_split

# Check class distribution before train-test split
print("Original class distribution in 'deceptive':")
print(y.value_counts())

# Train-test split with stratification to preserve class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verify that both training and test sets contain samples from each class
print("\nClass distribution in 'y_train':")
print(y_train.value_counts())
print("\nClass distribution in 'y_test':")
print(y_test.value_counts())

# Model training and evaluation
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Logistic Regression Model (if class imbalance is resolved)
if y_train.nunique() > 1:  # Ensure there is more than one class in y_train
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    print("\nLogistic Regression Model")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("Classification Report:\n", classification_report(y_test, y_pred_lr))
else:
    print("\nSkipping Logistic Regression training: y_train contains only one class.")
