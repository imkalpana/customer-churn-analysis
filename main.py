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

print("===========================================  All Columns  ===============================================\n")
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
print("=========================================  EDA  =================================================\n")


# Univariate Analysis

churn = df.groupby('Churn').size()
print(churn)

# churning rate
churn_rate = churn.iloc[1] / churn.sum()
print(f"Churn Rate: {churn_rate:.2%}")


# Tenure distribution
sns.histplot(df['tenure'], kde=True)
plt.title('Tenure Distribution')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.savefig('tenure_distribution.png')
plt.close()

'''
Shows that most customers have a tenure of just a few months, with a significant drop-off after that point.
This indicates that many customers leave the service shortly after joining. That might be due to recent
marketing campaigns or promotions attracting new customers who then quickly churn.
'''


# Monthly Charges distribution
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charges')
plt.ylabel('Number of Customers')
plt.savefig('monthly_charges_distribution.png')
plt.close()

# Total Charges distribution
sns.histplot(df['TotalCharges'], kde=True)
plt.title('Total Charges Distribution')
plt.xlabel('Total Charges') 
plt.ylabel('Number of Customers')
plt.savefig('total_charges_distribution1.png')
plt.close()

'''
most customers pay low monthly charges, but there is a great fraction with medium values. 
Since most customers have been with the company for just a few months, the total charges plot shows most 
customers with low values.
'''
# Bivariate Analysis


# Contract Type distribution
contract = df.groupby('Contract').size()
print(contract)


# Tenure vs. Contract
sns.histplot(x='tenure', hue='Contract', data=df, kde=True)
plt.title('Tenure distribution by Contract Type')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.savefig('tenure_by_contract.png')
plt.close()
'''
Customers with month-to-month contracts tend to have shorter tenures, while those with one-year and two-year contracts have longer tenures.
This suggests that longer-term contracts may help retain customers for a longer period.
'''

# Plot churn rate by contract type
churn_rate_by_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
print(churn_rate_by_contract)


churn_rate_by_contract['Yes'].plot(kind='bar')
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate')
plt.ylim(0, 0.5)
for idx, val in enumerate(churn_rate_by_contract['Yes']):
    plt.text(idx, val + 0.01, f"{val:.1%}", ha='center')
plt.tight_layout()

plt.savefig('churn_rate_by_contract.png')
plt.close()
'''
Customers with month-to-month contracts have a significantly higher churn rate around 43%
compared to those with one-year (11%) or two-year contracts (7%).
'''



#Churning trends by internet service type
churn_rate_by_internet = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
print(churn_rate_by_internet)

churn_rate_by_internet['Yes'].plot(kind='bar', color=['skyblue', 'orange', 'lightgreen'])
plt.title('Churn Rate by Internet Service Type')
plt.xlabel('Internet Service Type')
plt.ylabel('Churn Rate')
plt.ylim(0, 0.5)
for idx, val in enumerate(churn_rate_by_internet['Yes']):
    plt.text(idx, val + 0.01, f"{val:.1%}", ha='center')
plt.tight_layout()

plt.savefig('churn_rate_by_internet.png')
plt.close()
'''
Customers with Fiber optic internet service have a significantly higher churn rate around 42% 
compared to those with DSL (19%) or no internet service (7%).
'''

#Churn rate by payment method
churn_rate_by_payment = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
print(churn_rate_by_payment)

churn_rate_by_payment['Yes'].plot(kind='bar', color=['skyblue', 'orange', 'lightgreen'])
plt.title('Churn Rate by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Churn Rate')
plt.ylim(0, 0.5)
for idx, val in enumerate(churn_rate_by_payment['Yes']):
    plt.text(idx, val + 0.01, f"{val:.1%}", ha='center')
plt.tight_layout()

plt.savefig('churn_rate_by_payment.png')
plt.close()
'''
Customers who pay by electronic check have a significantly higher churn rate around 41% 
compared to those who pay by mailed check (15%) or credit card (10%).
'''

#Multivariate Analysis

# Churn by monthly charges and tenure

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='tenure',
    y='MonthlyCharges',
    hue='Churn',
    data=df,
    alpha=0.6,
    palette={'Yes': 'red', 'No': 'green'}
)
plt.title('Churn by Monthly Charges and Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Monthly Charges')
plt.legend(title='Churn')
plt.savefig('churn_by_monthlycharges_tenure.png')
plt.close()

'''
High monthly charges + short tenure = highest churn risk

'''


#Senior citizens churn rate











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
