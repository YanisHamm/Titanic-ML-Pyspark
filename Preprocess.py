from SparkSession import train
from SparkSession import *

train = train.fillna({'Embarked':'S'})
titles = {'Mr':'Mr', 'Miss':'Miss', 'Mrs':'Mrs', 'Master':'Master', \
             'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',\
             'Don': 'Mr', 'Mme': 'Miss', 'Jonkheer': 'Mr', 'Lady': 'Mrs',\
             'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs', \
             'Dr':'Mr', 'Rev':'Mr'}

map = create_map([lit(x) for x in chain(*titles.items())])
train = train.withColumn('Title', map[train['Title']])

def age_imputer(data, title, age):
    return data.withColumn('Age', when((data.Age.isNull()) & (data.Title==title),age).otherwise(data.Age))
   
train = age_imputer(train, 'Miss', 21.86)
train = age_imputer(train, 'Master', 4.75)
train = age_imputer(train,'Mr', 33.02)
train = age_imputer(train, 'Mrs', 35.98)

train = train.withColumn('FamilySize', train.Parch + train.SibSp).drop('Parch', 'SibSp')

train = train.drop('PassengerId','Cabin','Name','Ticket','Title')