#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:32:29 2023

@author: jialinshang
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter
#pd.set_option('display.max_columns', 6)

Contacts = pd.read_excel('/Users/jialinshang/Desktop/applica/Reports for Applica/Applicant_Master 7.12.23 Inactive and active.xlsx')
Contacts1 = Contacts.copy()

Policy =  pd.read_excel('/Users/jialinshang/Desktop/applica/Reports for Applica/Policy_Master 7.12.2023 Active and Inactive with all data available.xlsx')
Policy1 = Policy.copy()

Contacts1.drop(columns=['Branch Name', 'Account Name', 'Address - Line1',
       'Address - City', 'Address - Zip ', 'Email  - Primary',
       'Assigned Producer', 'CSR', 'Phone - Home ',
       'Relation', 'Account Type', 'Active Policy Count',
       'Active Policy Premium', 'Address - Line2', 'Address - Unit Number',
       'Applicant Type', 'Auto Carrier *', 'Auto Policy Number *',
       'Auto Policy Sold Date *', 'Auto Policy Term', 'Auto Premium',
       'Auto Renewal Date', 'Business Name', 'Commercial  SIC Code', 'County',
       'Created Date', 'Customer Since Date', 'Email - Alternate',
       'Email - Business', 'Email OptOut - Alternate',
       'Email OptOut - Business', 'Email OptOut - Primary', 'First Name',
       'Has Policy', 'Home Carrier *', 'Home Policy Number *',
       'Home Policy Sold Date *', 'Home Policy Term', 'Home Premium',
       'Home Renewal Date', 'Inactive Policy Count', 'Inception Date',
       'Is Commercial', 'Is Package', 'Is VIP', 'Last Modified Date',
       'Last Name', 'Lead Priority', 'Lead Source', 'Lead Status',
       'Nickname', 'Phone - Business',
       'Phone - Cell', 'Phone - Work', 'Policy Count', 'Probability Of Sale',
       'Quote Count', 'Address -  Zip Plus 4'], inplace=True)

Policy1.drop(columns=['Account Name', 'Account Type', 'Assigned Producer',
       'CSR', 'Branch Name', 'Phone - Home','Primary Email','Policy Expiration Date',
       'Policy Term', 'Policy Type','Master Company','Policy Number ',
       'Premium - Written', 'Producer Code', 'Rating State', 'Service Team',
       'Applicant Status', 'Transaction Type ', 'Address - City', 
       'Address- Line1', 'Address - Line2', 'Address - State', 'Premium - Annualized',
       'Address - Unit Number', 'Address - Zip Code', 'Agency Code',
       'Applicant Is VIP', 'Billing Company', 'Billing Type', 'Business Name',
       'Business Phone', 'Commission - Agency ', 'Customer Date Created',
       'Customer Since', 'Department', 'Email - Alternate', 'Email - Business',
       'Email  Opt Out - Alternate ', 'Email Opt Out - Business',
       'Email Opt Out - Primary', 'First Name', 'Last Name', 'Lead Source',
       'LOB Code','LOB Origination Date', 'NAICS Code',
       'Nick Name', 'Phone - Cell', 'Phone - Work', 'Policy Cancellation Date',
       'Policy Change Date', 'Policy Creation Date', 'Policy Description',
       'Policy Download Date', 
       'Policy Last Modified Date', 'Policy Master ID', 'Policy Status',
       'Premium - Changed', 'Premium - Estimated Fees',
       'Premium - Estimated Taxes', 'Premium - Full Term',
       'Premium - Percent Change', 'Producer Code Override', 'Rewrite Status',
       'Service Team Override', 'Source', 'Sub LOB', 'Transaction Type Code',
       'Writing Company Name'], inplace=True)

merge_df = pd.merge(Policy1, Contacts1, on="Applicant ID", how="inner")
merge_df.loc[merge_df['LOB Is Motorcycle'] == 'Yes', 'Line Of Business'] = 'Motorcycle'
df_cleaned = merge_df.drop(['LOB Is Motorcycle'],axis=1)
df_cleaned.replace('Unknown', pd.NA, inplace=True)

def preprocessing_pipeline(df):
    df = df.drop_duplicates()
    threshold = 0.8 * len(df.columns)
    df = df.dropna(thresh=threshold)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['Policy Effective Date'] = pd.to_datetime(df['Policy Effective Date']).dt.strftime('%Y-%m-%d')
    df['Gender'] = df['Gender'].str.strip().str.capitalize()
    df['Income'] = df['Income'].str.replace(',', 'X').str.replace('.', ',').str.replace('X', '.')
    df['Income'] = df['Income'].str.replace(',', '.').astype(float)
    return df

df_cleaned = preprocessing_pipeline(df_cleaned)
#%%
df_cleaned1 = df_cleaned.copy()
df_cleaned1['purchase_num'] = df_cleaned1.groupby('Applicant ID').cumcount() + 1
df_cleaned1['purchase_num'] = 'purchase ' + df_cleaned1['purchase_num'].astype(str)

# Pivot the table
pivot_df = df_cleaned1.pivot(index='Applicant ID', columns='purchase_num', values='Line Of Business').reset_index()
pivot_df1 = pivot_df.copy()


filtered_df = pivot_df1[~((pivot_df1['purchase 1'] == 'Auto (Personal)') & (pivot_df1['purchase 2'].isna()))]
filtered_df1 = filtered_df.copy()

# Loop through the rows and columns to identify duplicates
for i, row in filtered_df1.iterrows():
    seen_values = set()  # Initialize an empty set to keep track of unique values within the row
    for j in range(1, 10):  # Loop through each 'purchase' column (assuming there are 9)
        col_name = f'purchase {j}'
        
        value = row[col_name]
        
        if pd.isna(value):
            continue
            
        if value in seen_values:
            filtered_df1.at[i, col_name] = np.nan  # Set duplicate values to NaN
        else:
            seen_values.add(value)  # Add unique values to the set

print(filtered_df1)

melted_df = filtered_df1.melt(id_vars=['Applicant ID'],
                    value_vars=[f'purchase {i}' for i in range(1, 10)], 
                    var_name='purchase_num',
                    value_name='Line Of Business')

melted_df = melted_df.dropna(subset=['Line Of Business']).reset_index(drop=True)

melted_df['Line Of Business'].value_counts()
melted_df = melted_df.drop(columns=['purchase_num'])
df_cleaned2 = df_cleaned.drop(columns=['Line Of Business'])
merge_new = pd.merge(melted_df, df_cleaned2, on="Applicant ID", how="left")
merge_new = merge_new.drop_duplicates()

#%%
merge_new1 = merge_new.copy()
new_df = merge_new1.replace({'Line Of Business': {'Notary Bond': 'Bond','Dishonest Bond': 'Bond','Title Bonds': 'Bond', 'Bonds Miscellaneous':'Bond',
                                                  'Watercraft (small boat)': 'Boat Insurance', 'Life - Term': 'Life Insurance',
                                                  'Recreational Veh':'RV Insurance', 'Dwelling fire':'Homeowners','Mobile Homes':'Homeowners',
                                                  'Health':'Medicare','Short Term Medical':'Medicare','Modular Home':'Homeowners', }})

columns_to_rename = ['Inland Marine (pers)', 'Personal pkg', 'Vacant Property', 'Special Events', 'Roadside Assistance',
                     'Miscellaneous', 'FarmOwner', 'Business Owners Policy', 'Non-Owned', 'Sched Prpty',
                     'Cyber Liability', 'Vacant Property']

# Rename the columns
for col in columns_to_rename:
    new_df.rename(columns={col: 'Commercial Liability'}, inplace=True)
    
new_df['Commercial Liability'] = new_df[columns_to_rename].sum(axis=1)
new_df.drop(columns=columns_to_rename, inplace=True)

#%%
categ = ['Gender','Marital Status','Address - State']
df_encoded = pd.get_dummies(merge_new, columns = categ)
X = df_encoded.drop(['Line Of Business','Applicant ID'],axis=1)
y = df_encoded[['Line Of Business','Applicant ID']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

kmeans = KMeans(n_clusters=7, random_state=42)
shuffle = ShuffleSplit(n_splits=10, test_size=.25)
def custom_silhouette(estimator, X_train):
    labels = estimator.predict(X_train)
    return silhouette_score(X_train, labels)

results = cross_validate(kmeans, X_train, scoring=custom_silhouette, cv=shuffle, return_train_score=True)

# Output the results
print("Train silhouette scores:", results['train_score'])
print("Test silhouette scores:", results['test_score'])

kmeans.fit(X_train)

X_test['Cluster'] = kmeans.predict(X_test)
result = pd.concat([X_test, y_test], axis=1)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
    
# Plot the Elbow graph
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

N = 5
def top_n_frequent(s, n=N):
    return s.value_counts().head(n).index.tolist()

# Group by 'group_col' and apply the function to 'value_col'
top_values = result.groupby('Cluster')['Line Of Business'].apply(top_n_frequent, n=N)

print(top_values)
#%%
top_values = top_values.to_frame()
# Convert the 'Line Of Business' column into a list of lists
list_of_purchases = top_values['Line Of Business'].to_list()

# Determine the maximum length of lists in 'Line Of Business'
max_len = max(len(lst) for lst in list_of_purchases)

# Generate column names dynamically
col_names = [f'top purchase {i+1}' for i in range(max_len)]

# Convert lists to DataFrame with dynamic column names
df_split = pd.DataFrame(list_of_purchases, columns=col_names, index=top_values.index)
df_split = df_split.reset_index()

result = pd.concat([X_test, y_test], axis=1)
result = pd.merge(result, df_split, on="Cluster", how="left")

# Function to get top 2 purchases that are not equal to 'Line Of Business'
def get_top_purchases(row):
    # Filter purchases that are not equal to 'Line Of Business'
    valid_purchases = [p for p in row[col_names] if p != row['Line Of Business']]
    
    # Get top 2 valid purchases (assuming the purchases are already sorted; if not, you should sort them here based on your criteria)
    top_purchases = valid_purchases[:2]
    
    return pd.Series(top_purchases, index=['Top Purchase 1', 'Top Purchase 2'])

# Apply the function row-wise and concatenate the result with the original DataFrame
top_purchases_df = result.apply(get_top_purchases, axis=1)
df_result = pd.concat([result, top_purchases_df], axis=1)
df_result = df_result.drop(columns=col_names)
df_result_f = df_result[['Applicant ID','Line Of Business','Top Purchase 1','Top Purchase 2']]

r = 2
unique_values = Policy1['Line Of Business'].unique()
def get_random_values(line_of_business, unique_values):
    available_choices = [val for val in unique_values if val != line_of_business]
    return np.random.choice(available_choices, size=r, replace=False)

# Apply function row-wise and assign random values to 'Random1' and 'Random2'
r_names = [f'Random{i+1}' for i in range(r)]
df_result_f[r_names] = df_result_f.apply(lambda row: pd.Series(get_random_values(row['Line Of Business'], unique_values)), axis=1)

print(df_result_f.head())

df_result_f.to_excel('output_kmeans.xlsx', index=False, sheet_name='output_kmeans')