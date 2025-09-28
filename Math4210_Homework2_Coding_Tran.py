#Math 4210 - Homework 2
#Tran Gia Bao
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns       
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#Programming Problem 1:
#Part A - create binary variable that contains a 1 if mpg is above the median and a 0 if mpg is below the median
df = pd.read_csv('Auto.csv',na_values='?') 
df = df.dropna()#drop rows with missing values
df = df.apply(pd.to_numeric, errors='ignore') # Convert all columns to numeric where possible
median_mpg = df['mpg'].median()
df['mpg01'] = (df['mpg'] > median_mpg).astype(int)

#Part B - Explore the data with scatterplots and boxplots
# Scatterplot matrix
#sns.pairplot(df, y_vars = ['mpg01'], x_vars = [c for c in df.columns if c!= 'mpg01'],kind='scatter') #see the relationship between each pair
#plt.show()

#Part C - Split the data into a training set and a test set
traindf, test_df = train_test_split(df, test_size=0.2, random_state=42) #80% training and 20% test because relatively small dataset - would normally go for 70/30
predictors = ['displacement', 'weight', 'acceleration']
X_train = traindf[predictors]
y_train = traindf['mpg01']
X_test = test_df[predictors]
y_test = test_df['mpg01']

#Perform LDA on the training data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

#Test data prediction & accuracy calculation (LDA)
y_pred = lda.predict(X_test)
accuracy_lda = accuracy_score(y_test, y_pred)
test_error_lda = 1 - accuracy_lda
print(f'Test error of the LDA model: {test_error_lda:.4f}')

#Perform QDA on the training data
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

#Test data prediction & accuracy calculation (QDA)
y_pred_qda = qda.predict(X_test)
accuracy_qda = accuracy_score(y_test, y_pred_qda)
test_error_qda = 1 - accuracy_qda
print(f'Test error of the QDA model: {test_error_qda:.4f}')

#Perform Logistic Regression on the training data
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Test data prediction & accuracy calculation (Logistic Regression)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
test_error_logreg = 1 - accuracy_logreg
print(f'Test error of the Logistic Regression model: {test_error_logreg:.4f}')

#Perform Naive Bayes on the training data
nb = GaussianNB()
nb.fit(X_train, y_train)

#Test data prediction & accuracy calculation (Naive Bayes)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
test_error_nb = 1 - accuracy_nb
print(f'Test error of the Naive Bayes model: {test_error_nb:.4f}')

#Perform KNN on the training data with different K values.
k_values = [1, 2, 3, 4, 5, 7, 9]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    test_error_knn = 1 - accuracy_knn
    print(f'Test error of the KNN model with k={k}: {test_error_knn:.4f}')

#Programming Problem 2: 
dfBos = pd.read_csv('Boston.csv', na_values='?')
dfBos = dfBos.dropna()  # drop rows with missing values
dfBos = dfBos.apply(pd.to_numeric, errors='ignore')  # Convert all columns to numeric where possible
median_crim = dfBos['crim'].median()
dfBos['crim01'] = (dfBos['crim'] > median_crim).astype(int)

sns.pairplot(dfBos, y_vars_bos = ['crim01'], x_vars_bos = [col for col in dfBos.columns if col!= 'crim01'],kind='scatter')
plt.show()

train_bos, test_bos = train_test_split(dfBos, test_size=0.2, random_state=42)

# Select predictors for Boston data (example: 'nox', 'rm', 'age', 'dis', 'tax')
predictors_bos = ['nox', 'rm', 'age', 'dis', 'tax']
X_train_bos = train_bos[predictors_bos]
y_train_bos = train_bos['crim01']
X_test_bos = test_bos[predictors_bos]
y_test_bos = test_bos['crim01']

# Logistic Regression
logreg_bos = LogisticRegression()
logreg_bos.fit(X_train_bos, y_train_bos)
y_pred_logreg_bos = logreg_bos.predict(X_test_bos)
test_error_logreg_bos = 1 - accuracy_score(y_test_bos, y_pred_logreg_bos)
print(f'Boston Logistic Regression test error: {test_error_logreg_bos:.4f}')

# LDA
lda_bos = LinearDiscriminantAnalysis()
lda_bos.fit(X_train_bos, y_train_bos)
y_pred_lda_bos = lda_bos.predict(X_test_bos)
test_error_lda_bos = 1 - accuracy_score(y_test_bos, y_pred_lda_bos)
print(f'Boston LDA test error: {test_error_lda_bos:.4f}')

# Naive Bayes
nb_bos = GaussianNB()
nb_bos.fit(X_train_bos, y_train_bos)
y_pred_nb_bos = nb_bos.predict(X_test_bos)
test_error_nb_bos = 1 - accuracy_score(y_test_bos, y_pred_nb_bos)
print(f'Boston Naive Bayes test error: {test_error_nb_bos:.4f}')

# KNN for several k values
k_values_bos = [1, 3, 5, 7, 9]
for k in k_values_bos:
    knn_bos = KNeighborsClassifier(n_neighbors=k)
    knn_bos.fit(X_train_bos, y_train_bos)
    y_pred_knn_bos = knn_bos.predict(X_test_bos)
    test_error_knn_bos = 1 - accuracy_score(y_test_bos, y_pred_knn_bos)
    print(f'Boston KNN test error (k={k}): {test_error_knn_bos:.4f}')

#AI Statement: The majority of code generated was written by me. I did consult CoPilot and ChatGPT on how to implement certain functions within the SKLearn library.