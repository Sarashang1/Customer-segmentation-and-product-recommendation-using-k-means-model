#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 09 23:43:32 2023

@author: jialinshang
"""

import requests
import pandas as pd

pageList = range(1, 235) 

iterationComplete = False
allRecords = []
next_page_token = None
for page in pageList:
    if not iterationComplete:
        url = "https://www.zohoapis.com/crm/v4/Contactsss"
        headers = {
    "Authorization": "Zoho-oauthtoken 1000.0de48f745843fa7e3c8e9fc315eb1030.d47e690da5ffd387330bf1aded80a31683787" 
}
        params = {
    "fields": "Applicant_ID,Date_of_Birth,Education,Marital_Status,Gender,Occupation,Age，Income, Address - State", 
    "page_token": next_page_token
}

        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        records = data["data"]
        allRecords.append(records)
        next_page_token = data["info"]["next_page_token"]
        if not data["info"]["more_records"]:
            iterationComplete = True

flattened_data = [item for sublist in allRecords for item in sublist]
data = pd.DataFrame(flattened_data)

data.to_excel('contacts_0906.xlsx')

pageList = range(1, 235) 

iterationComplete = False
allRecords = []
next_page_token = None
for page in pageList:
    if not iterationComplete:
        url = "https://www.zohoapis.com/crm/v4/Leads"
        headers = {
    "Authorization": "Zoho-oauthtoken 1000.0de48f745843fa7e3c8e9fc315eb1030.d47e690da5ffd387330bf1aded80a31683787" 
}
        params = {
    "fields": "Applicant_ID, Line Of Business, Term_Effective_Date, Term_Expiration_Date", 
    "page_token": next_page_token
}

        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        records = data["data"]
        allRecords.append(records)
        next_page_token = data["info"]["next_page_token"]
        if not data["info"]["more_records"]:
            iterationComplete = True

flattened_data = [item for sublist in allRecords for item in sublist]
data = pd.DataFrame(flattened_data)

data.to_excel('policy_0906.xlsx')