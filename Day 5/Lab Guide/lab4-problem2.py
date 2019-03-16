import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'''
Step 1: Read the dataset using pandas.
'''

pokemon_dataset = pd.read_csv('data/pokemon.csv')

'''
Step 2: Access a certain groups of columns using pandas for preparing (X, y). 
Suppose that we want to have data according to the following columns:
- sp_attack
- sp_defense
- attack
- defense
- speed 
- hp
- type1
'''

# If we browse pokemon_dataset['type2'] in python console, we will see that many of them are null.
# What does this information tell us? This says a pokemon may be belonged to two types.
# Suppose that, in this example, we want to consider only pokemons which have a single type.
# How to handle this in pandas? (See https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.isnull.html)
# Pandas also provide a method called 'loc' to access a certain groups of rows and columns.

dataframe = pokemon_dataset[pokemon_dataset['type2'].isnull()].loc[
    :, ['sp_attack', 'sp_defense', 'attack', 'defense', 'speed', 'hp', 'type1']
]

# Grap only 'sp_attack', ..., 'hp' as an input X
# To index by position in pandas, see https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html
X = dataframe.iloc[:, :-1].values # เอาทุก column ไม่เอา column สุดท้าย

# Normalizing is not necessary for the classification; but, it will make visualizing task (easier for our eyes).
# So, let's do this since we will also visualize it at the end of this exercise !
# Noted that we will learn why normalizing can help visualizing later in this course ! (e.g. when it comes to PCA)
X_normalized = normalize(X)

# Grap the last column as a target y
y = dataframe.iloc[:, -1].values

'''
Step 3: Fit linear discriminant analysis model according to the given training data. 
See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
'''

linearDiscriminantAnalysis = LinearDiscriminantAnalysis()
# Try to read the document given above by yourself to train the mode
linearDiscriminantAnalysis.fit(X_normalized, y) # Now, we have trained with LDA

'''
Step 4: Show the predicted type for each pokemon and measure the accuracy. 
To predict class labels for samples in X, use method 'predict'
'''

# Try to read the document given in step 3 to make prediction

# After that, write codes to: 
# 1) show the predicted type of each pokemon
# 2) show the actual pokemon of each pokeon
# 3) show numerical value representing its accuracy
# Noted that there may be more than one line of code for this step 

predicted_type = linearDiscriminantAnalysis.predict(X_normalized)
print("Predicted type of each pokemaon is :")
print(predicted_type, '\n')

print("Actual type of each pokamon is: ",y, '\n')

accuracy = sum(predicted_type == y)/ len(y)
print("Accuracy is {0:f}", accuracy)
