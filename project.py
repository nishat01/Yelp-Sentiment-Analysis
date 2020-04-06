import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

#resampling the imbalanced training data for better disctribution 
def upsample_min(df):
    '''
    This function increases the number of samples in the minority class
    by randomly duplicating examples from the same class with replacement
    '''
    df_positive = df[df['class'] == 'positive']
    df_negative = df[df['class'] == 'negative']
    df_neutral = df[df['class'] == 'neutral']

    df_neutral_upsampled = resample(df_neutral, 
         replace = True,    #sample with replacement
         n_samples = 12000, #to match negative class
         random_state = 42  #to have reproducible results over multiple runs 
    )
   
    # Combine with upsampled minority class
    df_upsampled = pd.concat([df_positive, df_negative, df_neutral_upsampled])
    return df_upsampled

def downsample_max(df):
    '''
    This function reduces the number of examples in the majority class 
    '''
    df_positive = df[df['class'] == 'positive']
    df_negative = df[df['class'] == 'negative']
    df_neutral = df[df['class'] == 'neutral']

    df_pos_downsampled = resample(df_positive, 
         replace = False,    #sample without replacement
         n_samples = 20000,  #downsample to 20,000 samples
         random_state = 42   #to have reproducible results over multiple runs 
    )

    # Combine with downsampled majority class
    df_downsampled = pd.concat([df_neutral, df_negative, df_pos_downsampled])
    return df_downsampled

def remove_stop_words(text):
    """
        This function pre-processes a string i.e. removes stop words, converts into lowercase
        and removes any non-alphabet characters
        text: a string
        return: modified initial string
    """
    stop = stopwords.words('english')
    text = ' '.join(word.lower() for word in text.split() if word not in stop)

    tokens = nltk.word_tokenize(text)
    result = []
    for a in tokens:
        if a.isalpha():
            #only adding the tokens which contain all alpha letters to the array result
            result.append(a) 
    result = " ".join(str(x) for x in result)
    return result

def load_data():
    '''
    This function loads the training dataset and performs
    pre-processing tasks and train/test split
    
    return: x_train: 'text' values used for training
            x_test: 'text' values used for testing accuracy of the model
            y_train: 'class' values used for training
            y_test: 'class' values used for testing accuracy of the model
    '''
    df = pd.read_csv("train3.csv")
    df['text'] = df['text'].apply(remove_stop_words)
    #df['text'] = df['text'].apply(lambda x: [' '.join(stemmer.stem(y) for y in x.split())])
    df = upsample_min(df)
    df = downsample_max(df)
    df = df.set_index('ID')
    df.to_csv('processed_train3.csv')
    
    x = df['text']
    y = df['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    return (x_train, x_test, y_train, y_test)

'''
Split the training dataset into train and test datasets which we will use to train
the classifier and test the accuracy of the model
'''
x_train, x_test, y_train, y_test = load_data()

def fit_classifier(x_train, y_train, classifier, param_search, n):
    '''
    This function trains a classifier model using a scikit-learn Pipeline
    and performs a parameter grid search with k-fold cross-validation if configured
    
    x_train: 'text' values used for training the model
    y_train: 'class' values used for training the model
    classifier: 'svm', 'logreg', 'rf', 'tree'
    param_search: True or False
    n: cross validation n-fold (int)
    return: trained classifier
    '''
    
    if classifier == 'svm':    
        parameters = {
            'sgd__max_iter': [500,2000,2000],
            'sgd__n_iter_no_change': [5, 10, 20],
            'sgd__class_weight': ('balanced', None)
        }
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(strip_accents='unicode', min_df=3, ngram_range=(1,2))),
            ('lr', SGDClassifier(n_jobs=-1))
        ])
        
    elif classifier == 'logreg':
        parameters = {
            'lr__max_iter':[ 1000,2000,10000],
            'lr__class_weight': ('balanced',None),
            'lr__solver': ('saga', 'newton-cg', 'lbfgs'),
            'lr__penalty': ('l2', 'elasticnet')                   
        }
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(strip_accents='unicode', min_df=3, ngram_range=(1,2))),
            ('lr', LogisticRegression(n_jobs=-1))
        ])
        
    elif classifier == 'rf':
        parameters = {
            'rf__n_estimators': [100,500,1000],
            'rf__criterion': ('gini', 'entropy')
        }
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(strip_accents='unicode',  min_df=3, ngram_range=(1,2))),
            ('rf', RandomForestClassifier(n_estimators=500, criterion='gini',n_jobs=-1))
        ])
    
    else:
        print("Invalid classifier option")
        return
    
    if param_search:
        # parameter grid search + cross validation
        model = GridSearchCV(pipe, parameters, n_jobs=-1, cv=n)
        model.fit(x_train, y_train)
    else:
        # cross validation only
        scores = cross_validate(pipe, x_train, y_train, cv=n, n_jobs=-1, return_estimator=True)
        best_score = scores['test_score'].tolist().index(max(scores['test_score']))
        model = scores['estimator'][best_score]
        
    return model

# Set to True to perform an exhaustive parameter grid search
param_grid_search = False # BEWARE: It will take a long time to train if set to TRUE

# Number of folds for cross validation
k_fold = 10

# Train the models
sgd = fit_classifier(x_train, y_train, 'svm', param_grid_search, k_fold)
logreg = fit_classifier(x_train, y_train, 'logreg', param_grid_search, k_fold)
rf = fit_classifier(x_train, y_train, 'rf', param_grid_search, k_fold)
#tree = fit_classifier(x_train, y_train,'tree', param_grid_search, k_fold)

# Define list of classes in the dataset
data_classes = ['positive', 'negative', 'neutral']

#Predict values using the test dataset from the 70/30 split from above
#and output the accuracy of each model

y_pred = sgd.predict(x_test)
print('Linear SVM accuracy is:  %.2f' % (accuracy_score(y_test, y_pred)*100))
print(classification_report(y_test, y_pred, target_names=data_classes))

y_pred = logreg.predict(x_test)
print('Logistic Regression accuracy is:  %.2f'  % (accuracy_score(y_test, y_pred)*100))
print(classification_report(y_test, y_pred, target_names=data_classes))

y_pred = rf.predict(x_test)
print('Random Forest accuracy is:  %.2f' % (accuracy_score(y_test, y_pred)*100))
print(classification_report(y_test, y_pred, target_names=data_classes))

'''
Preparing the test data
'''
test = pd.read_csv("test3.csv")
test['text'] = test['text'].apply(remove_stop_words)
test = test.set_index('ID')
test.to_csv('processed_test3.csv')
test['CLASS'] = ''
y_test = test['text']

'''
Testing the trained models on the test data
'''
svm_preds = sgd.predict(y_test)
logreg_preds = logreg.predict(y_test)
rf_preds = rf.predict(y_test)

'''
The predictions by each trained model are stored into .csv output files
'''
df_svm_pred = test
df_svm_pred['CLASS'] = svm_preds
df_svm_pred.to_csv('output_svm.csv')
#df_svm_pred['CLASS'].value_counts()

df_logreg_pred = test
df_logreg_pred['CLASS'] = logreg_preds
df_logreg_pred.to_csv('output_logreg.csv')
#df_logreg_pred['CLASS'].value_counts()

df_rf_pred = test
df_rf_pred['CLASS'] = rf_preds
df_rf_pred.to_csv('output_rf.csv')
#df_rf_pred['CLASS'].value_counts()

'''
Final predictions by the best model are formatted 
and stored into a .csv file
'''
df_rf_pred.index.names = ['REVIEW ID']
df_rf_pred = df_rf_pred.drop(columns=['text'])
df_rf_pred.to_csv("prediction.csv")

print("Best model based on the highest F1-score for each class \nand overall accuracy rate is Random Forest.")
print("Total reviews in the test dataset: ", len(df_rf_pred))
print("Number of reviews per class: \n", df_rf_pred['CLASS'].value_counts())

print("Ouput files generated: processed_train3.csv, processed_test3.csv,\n output_svm.csv, output_logreg.csv, output_rf.csv, prediction.csv")

df_rf_pred['CLASS'].value_counts().plot(kind='bar')