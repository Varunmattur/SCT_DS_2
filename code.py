import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data for Titanic dataset
data = pd.DataFrame({
    'PassengerId': range(1, 11),
    'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    'Pclass': [3, 1, 3, 1, 3, 3, 2, 3, 2, 1],
    'Name': [
        'Allen, Miss. Elisabeth Walton', 'Cumings, Mr. John Bradley',
        'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath',
        'Allen, Mr. William Henry', 'Moran, Mr. James',
        'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard',
        'Johnson, Miss. Eleanor Ileen', 'Williams, Mr. Charles Eugene'
    ],
    'Sex': ['female', 'male', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'male'],
    'Age': [29, 38, 26, 35, 54, None, 54, 2, 30, 40],
    'SibSp': [0, 1, 0, 1, 0, 0, 0, 3, 1, 0],
    'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Ticket': ['PC 17599', 'STON/O2. 3101282', '113803', '373450', 'A/5 21171', '330877', '17463', '349909', '347742', 'CA 2144'],
    'Fare': [71.2833, 53.1000, 8.0500, 11.1333, 7.9250, 8.4583, 8.4583, 21.0750, 11.1333, 13.0000],
    'Cabin': [None, 'C85', None, 'C123', None, None, 'B51 B53 B55', None, None, None],
    'Embarked': ['S', 'C', 'S', 'S', 'Q', 'S', 'S', 'S', 'S', 'C']
})

# Data Cleaning
data['Age'].fillna(data['Age'].median(), inplace=True)
data.drop(columns=['Cabin', 'Ticket'], inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Exploratory Analysis
# Survival count
plt.figure(figsize=(8, 4))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(8, 4))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

# Survival by Gender
plt.figure(figsize=(8, 4))
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
