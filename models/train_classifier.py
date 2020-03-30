#!/usr/bin/env python

# import all relevant libraries
import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import parallel_backend


def load_data(database_filepath):
    """
    Function to load in data from sql database and split into X and Y
    
    input:
    - database_filepath : filepath to database
    
    returns:
    - X, Y and column names which are the 36 categories in the dataset
    
    """

    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_data_cleaned", engine)
    # print(df.head())
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    column_names = Y.columns
    return X, Y, column_names


def tokenize(text):
    """
    Function to nomalize, tokenize and lemmmatize sentences into words
    
    input:
    - text : sentence to be tokenized
    
    returns:
    - list of tokens
    
    """

    text = re.sub(r'[^a-zA-Z0-9]', " ", text.lower())
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).strip() for word in words if word not in stopwords.words("english")]
    return tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Estimator class that returns True if the first word in any of the sentences is a Verb, and False otherwise
    
    input:
    - text : sentence to be tokenized and associated with a part of speech
    
    returns:
    - True if first word in any of the tokenized sentences is a Verb
    - False otherwise
    
    """

    def starting_verb(self, text):
        """
        Function to nomalize, tokenize,lemmmatize and append part of speech (pos) to words in a sentence
        Also checks if first word is a verb

        input:
        - text : sentence

        returns:
        - Pandas Dataframe of either True or False

        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            # print("\n", pos_tags)
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Function to apply starting_verb function to all rows of X

        input:
        - X : rows of sentences

        returns:
        - Pandas Dataframe of either True or False

        """
        return pd.DataFrame(pd.Series(X).apply(self.starting_verb))


def build_model():
    """
    Function to build ML pipeline and run GridSearchCV with a list of specified parameters
    **I reduced the parameters dictionary because of computation time,
        you can comment out some or add to it as necessary

    input:
    - None

    returns:
    - ML model

    """

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("text_pipeline", Pipeline([
                ("count_vect", CountVectorizer(tokenizer=tokenize)),
                ("tfidf_vect", TfidfTransformer()),
            ])),
            ("starting_verb", StartingVerbExtractor())
        ])),
        ("clf", MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        "features__text_pipeline__count_vect__min_df": [2, 5],
        "clf__estimator__learning_rate": [0.1, 1],
        "clf__estimator__n_estimators": [50, 100],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate model and print pandas Dataframe showing
    precision, recall, f1_score and accuracy across all 36 categories
    
    input:
    - model: pre-built model
    - X_test : test data input
    - Y_test : test data output
    - category_names : names of 36 message categories in the dataset

    returns:
    - None

    """

    y_pred = model.predict(X_test)
    precision = []
    recall = []
    accuracy = []
    f1_score = []
    for i, col in enumerate(category_names):
        # print(y_test.loc[:,col])
        res = classification_report(Y_test.loc[:, col], y_pred[:, i], output_dict=True)
        precision.append(res["weighted avg"]["precision"])
        recall.append(res["weighted avg"]["recall"])
        f1_score.append(res["weighted avg"]["f1-score"])
        accuracy.append(res["accuracy"])

        # print("\n",classification_report(y_test.loc[:, col], y_pred[:, i]))
    print(pd.DataFrame(data={"precision": precision, "recall": recall, "f1-score": f1_score, "accuracy": accuracy},
                       index=category_names))


def save_model(model, model_filepath):
    """
    Function to save model 
    
    input:
    - model: pre-built model
    - model_filepath : filepath to where model should be saved as a pickle file
   

    returns:
    - None

    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # run jobs in process-based parallelism on a single host
        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, Y_train)
            # print("\n", "The best parameters used are: ", model.best_params_)

            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
