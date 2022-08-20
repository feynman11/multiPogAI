from collections import defaultdict
import logging

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Currently we import a formated product list that only includes columns for the classifier to consider
# We need to change this so that the program can determine which columns to use
df_products_complete = pd.read_csv('data/completeProductData.csv', index_col='UPC')

# Get the number of products from the dataframe, need this to compare against number of distinct values
productCount = df_products_complete.shape[0]

# The aim is to convert df_products_complete so that is matches df_products, for this we need to scan all the columns and count unique values
for column in df_products_complete:
    distinctCount = df_products_complete[column].unique().size
    logging.debug("The count of unique values in column %i is %i",column, distinctCount)
    if (distinctCount == 1):
        logging.info("Dropping column %s", column)
        df_products_complete.drop(columns = column, axis = 1, inplace=True)

for column in df_products_complete:
    logging.debug(column)

 # Import formated data
#df_products = pd.read_csv('data/allProducts.csv', index_col='UPC')
df_products = df_products_complete
# df = pd.read_csv('data\data.csv', index_col='UPC')

# Create dictionary to store encoding
d = defaultdict(LabelEncoder)

# Encoding the variables in data to integer values
df_products_encoded = df_products.apply(lambda x: d[x.name].fit_transform(x))

# Import the training data, products on the current planograms
df_train = pd.read_csv('data/trainingData.csv', index_col='UPC')
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

logging.info("Accuracy from testing %f", accuracy_score(y_test, testPredictions))

# Create new classifier using all the data to train it
classifier = tree.DecisionTreeClassifier()
classifier.fit(x, y)

# Load products without a planograms, and convert text using dictionary
df_test = pd.read_csv('data/testingData.csv', index_col='UPC')
df_test_fit = df_test.apply(lambda x: d[x.name].transform(x))

# Predict returns a numpy array, convert to data frame using index from test data
df_predictions_encoded = pd.DataFrame(classifier.predict(
    df_test_fit), columns=['Planogram'], index=df_test_fit.index)
# Use the dictionary to revert the transform from int back to string
df_predictions = df_predictions_encoded.apply(
    lambda x: d[x.name].inverse_transform(x))

# Export results to CSV
df_predictions.to_csv('data/results.csv', sep=',')

# Check the UPCs in results that we have planogram information for
df_predictions_test = pd.merge(df_train['Planogram'].to_frame().rename(columns={'Planogram': 'SourcePlanogram'}), df_predictions, left_index=True, right_index=True)

df_predictions_test.to_csv('data/resultsTest.csv', sep=',')

# UPCs in the source and results now stores in df_predictions_test loop over and work out % correct
totalCount = 0
correctCount = 0
for index, row in df_predictions_test.iterrows():
    totalCount += 1
    if(row['SourcePlanogram'] == row['Planogram']):
        correctCount += 1

logging.info("Accuracy from Source to Results is %f", correctCount/totalCount*100)
