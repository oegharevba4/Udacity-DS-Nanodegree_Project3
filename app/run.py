import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

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
        Also checks if first word has a pos of verb

        input:
        - text : sentence

        returns:
        - Pandas Dataframe of either True or False

        """
        sentence_list = nltk.sent_tokenize(text)
        # print("/n", sentence_list)
        # sentence_list = [i for i in sentence_list if i]
        # print("/n", sentence_list)
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data_cleaned', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # distribution of message categories
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    Y_sum = Y.sum(axis=0)
    column_names = list(Y.columns)
    
    # Correlation of message categories on heatmap
    corr = Y.corr()
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Genre'
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=column_names,
                    y=Y_sum
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'height': 600,
                'margin': dict(b=220),
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Message Categories'
                }
            }
        },
        
        {
            'data': [
                Heatmap(
                    x=column_names,
                    y=column_names,
                    z=corr
                )
            ],

            'layout': {
                'title': 'Correlation of Message categories',
                'height': 1000,
                'margin': dict(b=220, l=150, r=150),
                'xaxis': {
                    'title': 'Message Categories'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()