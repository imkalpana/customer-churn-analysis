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

'''
print(df.head())
print(df.describe())
print(df.info())
print(df.duplicated().sum())
print(df['TotalCharges'].isnull().sum())
print(df.dtypes)


with pd.option_context('display.max_columns', 22):
    print(df.head())

print(df.dtypes)
'''

# Change the datatype of 'SeniorCitizen' from int to object
df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})

# Change the datatype of 'TotalCharges' from object to float
# Convert non-numeric entries to NaN using 'coerce'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print(df.dtypes)
print(df.head())
print(df.info())
print(df['TotalCharges'].isnull().sum())



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
