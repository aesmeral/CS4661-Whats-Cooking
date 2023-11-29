import numpy as np
import pandas as pd
import nltk, json, os, re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

# Initialize NLTK
nltk.download('stopwords')
stemmer = PorterStemmer()

# Define stopwords
cooking_stopwords = ["recipe", "cook", "cooking", "bake", "boil", "grill", "saute", "roast", "simmer", "fry", "stir", "season", "dish", "plate", "meal", "serve"]
standard_stopwords = set(stopwords.words("english"))
stop_words = standard_stopwords.union(cooking_stopwords)

# Load data
training_data = pd.read_json('./train.json')
testing_data = pd.read_json('./test.json')

# Efficiently preprocess data
def scrubbed_ingredient(ingredient):
    scrubbed = re.sub(r'[^a-zA-Z ]', '', ingredient).lower()
    scrubbed = re.sub(r'oz|crushed|crumbles|ground|minced|powder|chopped|sliced|boneless|skinless|fresh|frozen|homemade|instance|kraft|large|lean|lowfat|small|smoke|vegan', '', scrubbed).lstrip()
    scrubbed = " ".join([stemmer.stem(word) for word in scrubbed.split() if word not in stop_words])
    return scrubbed

# Efficiently build ingredient vocabulary
ingredients_set = set()
for ingredients_list in training_data['ingredients']:
    scrubbed_ingredients = [scrubbed_ingredient(ingredient) for ingredient in ingredients_list]
    scrubbed_ingredients = [word for word in scrubbed_ingredients if len(word) >= 1]
    ingredients_set.update(scrubbed_ingredients)

sorted_set = sorted(ingredients_set)

# Create a sparse matrix
vectorizer = CountVectorizer(binary=True, vocabulary=sorted_set)
X = vectorizer.transform([" ".join([scrubbed_ingredient(ingredient) for ingredient in row]) for row in training_data['ingredients']])

# Create a sparse DataFrame
df = pd.DataFrame.sparse.from_spmatrix(X)
df.columns = sorted_set
df['cuisine'] = training_data['cuisine']
df['id'] = training_data['id']

# Split data
X = df.iloc[:, :-2]
y = df['cuisine']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train and evaluate the model
model = LogisticRegression(multi_class='ovr', solver='sag', max_iter=10000)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)

# Save results
results = pd.DataFrame()
results['actual'] = y_test
results['prediction'] = y_predict
results.to_csv('results.csv', index=False)

print(accuracy)
print(results.shape)
