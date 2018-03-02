import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# Import formated data
df_products= pd.read_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/allProducts.csv', index_col='UPC')
# df = pd.read_csv('F:\Google Drive\Documents\Code\multiPogAI\data\data.csv', index_col='UPC')

# Create dictionary to store encoding
d = defaultdict(LabelEncoder)

# Encoding the variables in data to integer values
df_products_encoded = df_products.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
#df_fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
#df.apply(lambda x: d[x.name].transform(x))


df_train = pd.read_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/trainingData.csv', index_col='UPC')
df_train_encoded = df_train.apply(lambda x: d[x.name].transform(x))

x = df_train_encoded[df_train_encoded.columns.drop('Planogram')]
y = df_train_encoded['Planogram']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(x_train, y_train)

testPredictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, testPredictions))


df_test = pd.read_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/testingData.csv', index_col='UPC')
df_test_fit = df_test.apply(lambda x: d[x.name].transform(x))

# Predict returns a numpy array
df_predictions_encoded = pd.DataFrame(my_classifier.predict(df_test_fit), columns=['Planogram'])
df_predictions = df_predictions_encoded.apply(lambda x: d[x.name].inverse_transform(x))

df_test['Planogram'] = df_predictions['Planogram']
df_test.Planogram = df_test.Planogram.astype(float)
df_test.to_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/results.csv', sep=',')




