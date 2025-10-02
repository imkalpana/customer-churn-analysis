'''
import csv

with open('telco_customer_churn_dataset.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        print (row)

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('telco_customer_churn_dataset.csv')

# Initial Data Exploration
print("=================================  Initial Data Exploration  =================================\n")

print(df.head())
print(df.describe())
print(df.info())
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.dtypes)


# Display all columns

print("=================================  All Columns  =================================\n")
with pd.option_context('display.max_columns', 22):
    print(df.head())

# Change the datatype of 'SeniorCitizen' from int to object
df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})

# Change the datatype of 'TotalCharges' from object to float
# Convert non-numeric entries to NaN using 'coerce'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print(df.dtypes)
print(df.head())
print(df.info())
print(df.isnull().sum())

#Check rows with NaN in 'TotalCharges'
print(df.loc[df['TotalCharges'].isnull()])
print(df['TotalCharges'].describe())

df['TotalCharges'].hist()
plt.title('Total Charges Distribution')
plt.xlabel('Total Charges')
plt.ylabel('Frequency')
plt.savefig('total_charges_distribution.png')
plt.close()

# Fill NaN with median value
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print(df['TotalCharges'].isnull().sum())

# Check unique values in categorical columns
print("=================================  Unique Values in Categorical Columns  =================================\n")
print(df.nunique())

print(df.describe(include=['object']))

'''
# most customers
- male
- younger than 65
- have no partner and no dependents
- have phone service
- have fibre optic internet service
- do not have online services
- do not have tech support
- do not have streaming services
- have month-to-month contract
- do not have paperless billing and pay by electronic check

'''
print("=================================  Visualization  =================================\n")

churn = df.groupby('Churn').size()
print(churn)

# churning rate
churn_rate = churn[1] / churn.sum()
print(f"Churn Rate: {churn_rate:.2%}")

class ChurnPredictor:
    def __init__(self, dataframe):
        self.df = dataframe

    def show_summary(self):
        print("Summary Statistics:")
        print(self.df.describe(include='all'))

    def plot_churn_distribution(self):
        sns.countplot(x='Churn', data=self.df)
        plt.title('Churn Distribution')
        plt.savefig('churn_distribution.png')
        plt.close()

    def plot_churn_distribution1(self):
        churn_counts = self.df['Churn'].value_counts()
        churn_counts.plot(kind='bar')
        plt.title('Churn Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.savefig('churn_distribution1.png')
        plt.close()



#visualizer = ChurnPredictor(df)
#visualizer.show_summary()
#visualizer.plot_churn_distribution1()
