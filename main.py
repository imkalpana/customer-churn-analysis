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
'''



class ChurnPredictor:
    def __init__(self, dataframe):
        self.df = dataframe

    def show_summary(self):
        print("Summary Statistics:")
        print(self.df.describe(include='all'))

    def plot_churn_distribution(self):
        sns.countplot(x='Churn', data=self.df)
        plt.title('Churn Distribution')
        plt.show()

    def plot_churn_distribution1(self):
        if 'Churn' in self.df.columns:
            churn_counts = self.df['Churn'].value_counts()
            churn_counts.plot(kind='bar')
            plt.title('Churn Distribution')
            plt.xlabel('Churn')
            plt.ylabel('Count')
            plt.show()
        else:
            print("'Churn' column not found in dataset.")



visualizer = ChurnPredictor(df)
#visualizer.show_summary()
visualizer.plot_churn_distribution1()
