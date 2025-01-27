# importing the necessary libraries
from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') # downloading the english language corpus

# initialising the Flask App [1]
app = Flask(__name__)

# loading pre-trained CountVectorizer model [3]
with open('count_vectorizer.pkl', 'rb') as file:
    count_vectorizer = pickle.load(file)

# loading the pre-trained Logistic Regression model for classification [3]
with open("count_feature_model.pkl", 'rb') as file:
    model = pickle.load(file)

# loading the dataset
df = pd.read_csv("assignment3_II.csv")

# initialising the lemmatizer [4]
lemmatizer = WordNetLemmatizer()

# function to lemmatize words i.e. lemmatize reviews 
def lemmatize_search(word):
    lemmatized_word = lemmatizer.lemmatize(word)
    return lemmatized_word

# route to home page [1]
@app.route('/')
def index():
    return render_template('home.html') # rendering the home page html template


# route to search functionality [1] 
@app.route('/search', methods=['GET', 'POST']) 
def search():
    if request.method == 'POST':  # [6]
        # when user searches a category by typing in the search bar i.e. user POSTing the query
        search_string = request.form['searchword'].lower()
    else: #[7]
        # when user clicks on a category to browse the desired product i.e. GETting a request from user
        search_string = request.args.get('searchword', '').lower()

    # if user does not enter an item in search bar i.e. search string is empty
    if not search_string:
        error_message = "Please enter an item."
        return render_template('home.html', error_message=error_message) # render home page with error message

    search_terms = search_string.split()  # splitting search string into individual words
    lemmatized_terms = [lemmatize_search(term) for term in search_terms]  # lemmatizing each word [4]

    # creating a pattern that matches all words in the search query [4]
    search_pattern = '|'.join(lemmatized_terms)

    # filtering items based on the search keyword in "Clothes Title" and "Cothes Description" [8]
    results = df[df['Clothes Title'].str.lower().str.contains(search_pattern) |
                     df['Clothes Description'].str.lower().str.contains(search_pattern)]

    # removing duplicates from results
    unique_results = results.drop_duplicates(subset=['Clothing ID'])

    # counting the unique results to display the count
    num_results = len(unique_results)

    # rendering search html template 
    return render_template('search.html', num_results=num_results, search_string=search_string, results=unique_results.to_dict('records'))

# route to display details of a specific clothing item and its reviews
@app.route('/item/<int:item_id>')
def item_details(item_id):
    # getting clothing item details using 'Clothing ID'
    item = df[df['Clothing ID'] == item_id].iloc[0]

    # getting all the reviews for a particular 'Clothing ID'
    reviews = df[df['Clothing ID'] == item_id][['Review Text']]

    # rendering itemdetails html template
    return render_template('itemdetails.html', item=item, reviews=reviews)

# route to handle the submission of a new review
@app.route('/submit_review', methods=['POST'])
def submit_review():
    if request.method == 'POST': # [6]
        # getting the form inputs: Review Title, Review Text and Cothing ID [9]
        title = request.form['title']
        review_text = request.form['review_text']
        item_id = request.form['item_id']
        
        # transforming the review text using the count vectorizer [1]
        review_vec = count_vectorizer.transform([review_text])
        
        # predicting recommendation label (0 = Not Recommended, 1 = Recommended) [1]
        pred_label = model.predict(review_vec)[0]
        
        # after prediction, the user is shown the result with an option to override [1]
        return render_template('submitreview.html', pred_label=pred_label, title=title, review_text=review_text, item_id=item_id)
    
    # if the request method is not POST, return to the home page
    return render_template('home.html')

# route to confirm and save the review after the user gives final recommendation
@app.route('/confirm_review', methods=['POST'])
def confirm_review():
    # getting the form inputs: Review Title, Review Text, Cothing ID and final user recommendation [9]
    title = request.form['title']
    review_text = request.form['review_text']
    item_id = request.form['item_id']
    user_label = request.form['user_label']  

    # creating a dictionary to store new reviews; empty fields for columns not mentioned in the form
    new_review = {
        'Clothing ID': item_id,
        'Age':"",
        'Title': title,
        'Review Text': review_text,
        'Rating': "",
        'Recommended IND': user_label,
        'Positive Feedback Count': "",
        'Division Name': "",
        'Department Name': "",
        'Class Name': "",
        'Clothes Title': title,
        'Clothes Description': ""
    }

    # converting new review dictionary into dataframe
    new_df = pd.DataFrame([new_review])

    # appending the new review to the CSV file
    new_df.to_csv('assignment3_II.csv', mode='a', header=False, index=False)  

    # refreashing data hence setting it to global
    global df  

    # reloading the dataset with the new reviews [10]
    df = pd.read_csv("assignment3_II.csv") 

    # rendering confirmreview html page
    return render_template('confirmreview.html', user_label=user_label, item_id=item_id)

if __name__ == '__main__':
    app.run(debug=True)

"""
References:
[1] Canvas/Modules/Week11 - Lab/w11_example2_bbc_backend_fasttext/app.py
[2] Saving as pkl file: https://www.analyticsvidhya.com/blog/2021/08/quick-hacks-to-save-machine-learning-model-using-pickle-and-joblib/
[3] Opening pkl file: https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
[4] Lemmatization: https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
[5] Understanding GET & POST: https://www.geeksforgeeks.org/flask-http-methods-handle-get-post-requests/
[6] Search bar POST: https://medium.com/analytics-vidhya/how-to-build-a-simple-search-engine-using-flask-4f3c01fe80fa
[7] Search bar GET: https://www.geeksforgeeks.org/using-request-args-for-a-variable-url-in-flask/
[8] Filtering items: https://stackoverflow.com/questions/11350770/filter-pandas-dataframe-by-substring-criteria
[9] Getting inputs from form: https://stackoverflow.com/questions/12277933/send-data-from-a-textbox-into-flask
[10] Appending to csv: https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file
[11] Understanding deployment: https://www.analyticsvidhya.com/blog/2020/04/how-to-deploy-machine-learning-model-flask/
 https://medium.com/@sooryanarayan_5231/end-to-end-machine-learning-deployment-from-model-training-to-web-service-integration-using-flask-4263d96b9479

"""