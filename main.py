'''
import csv

with open('telco_customer_churn_dataset.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        print (row)

'''

import pandas as pd

df = pd.read_csv('telco_customer_churn_dataset.csv')
print(df)

# add a class to visualise to statistics 
