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
print(df.head())
print(df.describe())

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



visualizer = ChurnPredictor(df)
visualizer.show_summary()
visualizer.plot_churn_distribution()