from collections import defaultdict

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Import formated data
df_products= pd.read_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/allProducts.csv', index_col='UPC')
# df = pd.read_csv('F:\Google Drive\Documents\Code\multiPogAI\data\data.csv', index_col='UPC')

# Create dictionary to store encoding
d = defaultdict(LabelEncoder)

# Encoding the variables in data to integer values
df_products_encoded = df_products.apply(lambda x: d[x.name].fit_transform(x))

# Import the training data, products on the current planograms
df_train = pd.read_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/trainingData.csv', index_col='UPC')
# Use the dictionary to convert string values into ints
df_train_encoded = df_train.apply(lambda x: d[x.name].transform(x))

# Split the training data into values (x) and results (y)
x = df_train_encoded[df_train_encoded.columns.drop('Planogram')]
y = df_train_encoded['Planogram']

# First we'll split the data and run some testing to see the accuracy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# Create test classifier and train it with split training data
testClassifier = tree.DecisionTreeClassifier()
testClassifier.fit(x_train, y_train)
testPredictions = testClassifier.predict(x_test)

print("Accuracy from testing",accuracy_score(y_test, testPredictions),"%")

# Create new classifier using all the data to train it
classifier = tree.DecisionTreeClassifier()
classifier.fit(x,y)

# Load products without a planograms, and convert text using dictionary
df_test = pd.read_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/testingData.csv', index_col='UPC')
df_test_fit = df_test.apply(lambda x: d[x.name].transform(x))

# Predict returns a numpy array, convert to data frame using index from test data
df_predictions_encoded = pd.DataFrame(classifier.predict(df_test_fit), columns=['Planogram'], index=df_test_fit.index)
# Use the dictionary to revert the transform from int back to string
df_predictions = df_predictions_encoded.apply(lambda x: d[x.name].inverse_transform(x))

# Export results to CSV
df_predictions.to_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/results.csv', sep=',')

# Check the UPCs in results that we have planogram information for
df_predictions_test = pd.merge(df_train['Planogram'].to_frame().rename(columns = {'Planogram':'SourcePlanogram'}), df_predictions, left_index=True, right_index=True)

#df_predictions_test.to_csv('/Users/thomasseagrave/Google Drive/Documents/Code/multiPogAI/data/resultsTest.csv', sep=',')

#UPCs in the source and results now stores in df_predictions_test loop over and work out % correct