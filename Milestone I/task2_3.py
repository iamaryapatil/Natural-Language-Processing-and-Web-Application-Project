#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Arya Ramesh Patil
# #### Student ID: S4060675
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * numpy
# * CountVectorizer
# * TfidfVectorizer
# * gensim.downloader
# * train_test_split
# * LogisticRegression
# * KFold
# * RegexpTokenizer
# * sent_tokenize
# * chain
# * division
# 
# ## Introduction
# This part of the assessment focuses on generating feature representations for clothing reviews and using them to classify item recommendations. Task 2 involves creation of vector representations and word embeddings (weighted and unweighted) based on one of the said models. I have chosen pre-trained glove model (glove-wiki-gigaword-50). These representations capture the essential information from the reviews for further analysis. Task 3 involves building machine learning models to classify whether an clothing item is recommended based on a review. The task also involves, experimenting with 3 different types of feature representations (count feature, weighted and unweighted features) to determine which one performs best and to evaluate if adding extra information such as 'Review Title' improves classification accuracy. The column 'Review Title' is pre-processed the same way as 'Review Text' in Task 1. To get robust results, the classification is performed using Kfolds (where number of folds are set at 5 by default).

# ## Importing libraries 

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from __future__ import division
from nltk.probability import *


# ## Task 2. Generating Feature Representations for Clothing Items Reviews

# This task involves creating 3 models namely:
# * Count Vector Representation
# * Unweighted Embedding
# * Weighted Embedding
# 

# In[2]:


# assigning file name to csv_file variable
csv_file = 'processed.csv'


# In[3]:


# importing the said csv file into a dataframe
clothes_review_data = pd.read_csv(csv_file, sep = ',')


# In[4]:


# checking the data
clothes_review_data


# When saving the file as 'processed.csv' in Task 1, the tokens were not enclosed in quotations. However, when the file is opened in Excel, the tokens are automatically stored as strings, adding quotations around them. As a result, when reading the 'processed.csv' file for subsequent tasks in Task 2, the tokens appear with quotations. To work with the tokens without the quotations, I found through further research that the eval() function can be used to remove the quotes and restore the original token format.

# In[5]:


# removing quotations around the review tokens
clothes_review_data['Processed Review Text'] = clothes_review_data['Processed Review Text'].apply(eval) # [1]


# In[6]:


clothes_review_data # checking the data


# In[7]:


# extracting 'Processed Review Text' in 'review_text' variable
review_text = clothes_review_data['Processed Review Text']


# In[8]:


# loading the vocabulary file
vocab_file = "vocab.txt"


# 'vocab' is stored as a dictionary because dictionaries allow efficient mapping between words (keys) and their corresponding integer indices (values). This is useful for quickly looking up the index of a word when processing text. Moreover, the format of the content naturally fits the dictionary structure.

# In[9]:


vocab = {} # initialising an empty dictionary
with open(vocab_file, 'r') as f: # opening the vocabulary file in read mode
    for line in f: # looping through each line
        word, index = line.strip().split(':') # stripping the leading/trailing whitespaces and splitting on ':'
        vocab[word] = int(index) # reading it in said format 


# In[10]:


vocab


# In[11]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 47
cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer


# In[12]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 48
count_features = cVectorizer.fit_transform([' '.join(review) for review in review_text]) 


# In[13]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 48
print(count_features.shape)


# In[14]:


print(count_features)


# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[15]:


# saving the count vector representation in a file in '#index,word_integer_index:word_freq' format
with open("count_vectors.txt", "w") as count_file: # opening the file in write mode
    for i, review in enumerate(count_features): # looping thorough each review and its corresponding feature vector
        descp = ','.join(f"{idx}:{count_features[i, idx]}" for idx in review.indices) # geting the word index and its frequency
        count_file.write(f"#{i},{descp}\n") # writing it in the said format


# In[16]:


# [2] w08_act2_embedding_classification - cell 11
# loading the pre-trained glove model
preTW2v_wv = api.load('glove-wiki-gigaword-50')


# This function generates document vectors by summing the embeddings of valid words for each document. The dimensions of the matrix are set such that number of rows is number of documents and number of columns is corresponding embedding dimensions. Each document is represented as a single vector, which is the sum of the word embeddings.

# In[17]:


# [2] w08_act2_embedding_classification - cell 15
# function that generates document vectors by summing the embeddings of valid words
def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size)) # initialising empty matrix 
    for i, doc in enumerate(docs): # looping through each document and its index
        valid_keys = [term for term in doc if term in embeddings.key_to_index] # filtering words that exist in embeddings vocabulary
        docvec = np.vstack([embeddings[term] for term in valid_keys]) # stacking the words and creating a matrix
        docvec = np.sum(docvec, axis=0) # summing the embeddings along the first axis to get a single vector representing the document
        vecs[i,:] = docvec # assigning the computed document vector to the corresponding row in the matrix
    return vecs # returning the matrix


# In[18]:


# [2] w08_act2_embedding_classification - cell 16
unweighted_vec_rep = docvecs(preTW2v_wv, review_text)


# In[19]:


# [2] w08_act2_embedding_classification
unweighted_vec_rep


# In[20]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 51
# from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialising the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform([' '.join(review) for review in review_text]) # generating the tfidf vector representation for all reviews
tfidf_features.shape # checking the dimensions


# In[21]:


print(tfidf_features) # checking the data


# This function writes the TF-IDF vector representation of each document into a file

# In[22]:


# [4] Act 3_ Generating Feature Vectors.ipynb - cell 58
def write_vectorFile(tfidf_features,filename):
    num = tfidf_features.shape[0] # getting the number of documents
    out_file = open(filename, 'w') # opening the file in write mode
    for a_ind in range(0, num): # loop through each review 
        for f_ind in tfidf_features[a_ind].nonzero()[1]: # looping through non-zero elements
            value = tfidf_features[a_ind][0,f_ind] # retrieving the value of the entry 
            out_file.write("{}:{} ".format(f_ind,value)) # writing in the said format
        out_file.write('\n') # new line
    out_file.close() # closing the file


# In[23]:


# [4] Act 3_ Generating Feature Vectors.ipynb - cell 59
write_vectorFile(tfidf_features,"./reviews_tVector.txt") # writing the tfidf vector to file


# In[24]:


vocab


# In[25]:


# [2] w08_act2_embedding_classification.ipynb - 4. Generating TF-IDF weighted document vectors - cell 2
def doc_wordweights(fName_tVectors, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}  # Reversing the vocab dictionary to map indices to words for easy lookup
    tfidf_weights = [] # initialising empty list

    with open(fName_tVectors) as tVecf: # opening the tvector of file of reviews
        tVectors = tVecf.read().splitlines()  # reading the file and splitting on each line

    for tv in tVectors: # looping through each vector
        tv = tv.strip()
        weights = tv.split(' ')  # splitting on 'word_index:weight' entries

        # filtering out any empty or improperly formatted entries
        weights = [w.split(':') for w in weights if ':' in w and len(w.split(':')) == 2]

        # constructing the word-weight dictionary 
        wordweight_dict = {inv_vocab[int(w[0])]: float(w[1]) for w in weights}
        tfidf_weights.append(wordweight_dict)

    return tfidf_weights # returning dictionaries


# In[26]:


# [2] w08_act2_embedding_classification.ipynb - 4. Generating TF-IDF weighted document vectors - cell 2
# reading TF-IDF weights from the file
fName_tVectors = 'reviews_tVector.txt'
tfidf_weights = doc_wordweights(fName_tVectors, vocab)


# In[27]:


tfidf_weights[0]


# In[28]:


# [2] w08_act2_embedding_classification.ipynb - 4. Generating TF-IDF weighted document vectors - cell 5
def weighted_docvecs(embeddings, tfidf, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size)) # initialising empty matrix
    for i, doc in enumerate(docs): # looping through each document and its index
        valid_keys = [term for term in doc if term in embeddings.key_to_index] # filtering words that exist in embeddings vocabulary
        tf_weights = [float(tfidf[i].get(term, 0.)) for term in valid_keys] # retrieving weights for for valid keys
        
        assert len(valid_keys) == len(tf_weights) # ensuring number of keys match the number of weights
        
        # calculating the weighted vectors
        weighted = [embeddings[term] * w for term, w in zip(valid_keys, tf_weights)]
        docvec = np.vstack(weighted) if weighted else np.zeros((1, embeddings.vector_size))  # handling empty cases
        docvec = np.sum(docvec, axis=0) # summing to get final document vector
        vecs[i, :] = docvec # storing the vector
    return vecs # returning the array


# In[29]:


# [2] w08_act2_embedding_classification.ipynb - 4. Generating TF-IDF weighted document vectors - cell 6
weighted_preTW2v_dvs = weighted_docvecs(preTW2v_wv, tfidf_weights, review_text)


# In[30]:


weighted_preTW2v_dvs[0]


# ## Task 3. Clothing Review Classification

# Building classification models based on different document feature represetations. 

# In[31]:


# [2] w08_act2_embedding_classification.ipynb - 4. Generating TF-IDF weighted document vectors - cell 8
seed = 0 # setting a seed to make sure the output is reproducible


# In[32]:


# [2] w08_act2_embedding_classification.ipynb - 4. Generating TF-IDF weighted document vectors - cell 8
models = [count_features,unweighted_vec_rep,weighted_preTW2v_dvs]
model_names = ['Count Feature','Unweighted Pretrained Word2Vec', 'Weighted Pretrained Word2Vec']
for i in range(0,len(models)): #loop through each model
    dv = models[i]
    
    # creating training and test split
    X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(dv, clothes_review_data['Recommended IND'], list(range(0,len(clothes_review_data))),test_size=0.33, random_state=seed)

    model = LogisticRegression(max_iter = 2000,random_state=seed) # initialising the model
    model.fit(X_train, y_train) # training the model
    
    print(f"Accuracy of {model_names[i]}: ", model.score(X_test, y_test))


# In[33]:


# [5] Act 4_Document Classification.ipynb
labels = clothes_review_data['Recommended IND']
labels


# In[34]:


# [5] Act 4_Document Classification.ipynb - cell 36
num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialising a 5 fold validation
print(kf)


# In[35]:


# [5] Act 4_Document Classification.ipynb - cell 37
def evaluate(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(max_iter = 500, random_state=seed) # initialising the model
    model.fit(X_train, y_train) # training the model
    return model.score(X_test, y_test) # returning the accuracy score


# In[36]:


# [5] Act 4_Document Classification.ipynb - cell 38
num_models = 3
cv_df = pd.DataFrame(columns = ['Count','Unweighted','Weighted'],index=range(num_folds)) # creating a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(labels)))): # looping through each split
    # extracting labels based on current fold
    y_train = [labels[i] for i in train_index]
    y_test = [labels[i] for i in test_index]

    # Count Features Model Evaluation
    X_train_count, X_test_count = count_features[train_index], count_features[test_index]
    cv_df.loc[fold,'Count'] = evaluate(count_features[train_index],count_features[test_index],y_train,y_test,seed)

    # Unweighted Vector Model Evaluation
    X_train_unweighted, X_test_unweighted = unweighted_vec_rep[train_index], unweighted_vec_rep[test_index]
    cv_df.loc[fold,'Unweighted'] = evaluate(unweighted_vec_rep[train_index],unweighted_vec_rep[test_index],y_train,y_test,seed)

    # Weighted Vector Model Evaluation
    X_train_weighted, X_test_weighted = weighted_preTW2v_dvs[train_index], weighted_preTW2v_dvs[test_index]
    cv_df.loc[fold,'Weighted'] = evaluate(weighted_preTW2v_dvs[train_index],weighted_preTW2v_dvs[test_index],y_train,y_test,seed)
    
    fold +=1


# In[37]:


# [5] Act 4_Document Classification.ipynb - cell 39
cv_df # checking the accuracy score of each fold for each model


# ## Pre-processing the column 'Title'

# In[38]:


# [6]
# extracting the 'Title' column for pre-processing
title = clothes_review_data['Title']
title


# In[39]:


# [6]
# function for case normalisation and tokensisation
def tokenize_reviews(review_text):
    nl_review = review_text.lower()  # converting reviews to lowercase
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?" # regex pattern to tokenise the reviews
    tokenizer = RegexpTokenizer(pattern) # tokeniser to split the reviews based on the regex pattern
    tokenized_review = tokenizer.tokenize(nl_review) # passing normalised reviews to get list of tokens
    return tokenized_review # returning cleaned list of tokens


# In[40]:


# [6]
tokenized_title = title.apply(tokenize_reviews) # applying the function to get cleaned list of tokens
tokenized_title


# In[41]:


# [6]
#loading the stopwords file
stopwords_file = "stopwords_en.txt"


# In[42]:


# [6]
# reading the file 
with open(stopwords_file, 'r') as file: # opens file in read mode
    stopwords = file.read().splitlines() # splits into line such that there is one stopwword per line


# In[43]:


# [6]
# getting unique stopwords
stopwords = set(stopwords)
len(stopwords)


# In[44]:


# [6]
# function for removal of words
def remove_words(tokens, stopwords):
    tokens = [token for token in tokens if len(token) >= 2] # removing words with the length less than 2
    tokens = [token for token in tokens if token not in stopwords] # removing stopwords 
    return tokens # returning cleaned list of tokens


# In[45]:


# [6]
# applying the function on tokenized reviews to remove the words on said condition
cleaned_title = tokenized_title.apply(lambda tokens: remove_words(tokens, stopwords))
cleaned_title


# In[46]:


# [6]
# [3] w07_act1_gen_feat_vec.ipynb - cell 11
words = list(chain.from_iterable(cleaned_title)) # we put all the tokens in the corpus in a single list
vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words


# In[47]:


# [6]
# [3] w07_act1_gen_feat_vec.ipynb - cell 12
term_fd = FreqDist(words) # compute term frequency for each unique word/type
term_fd


# In[48]:


# [6]
# removing the words that only appear once in the document collection
term_freq = cleaned_title.apply(lambda tokens: [word for word in tokens if term_fd[word] > 1])
term_freq


# In[49]:


# [6] 
# [3] w07_act1_gen_feat_vec.ipynb - cell 15
words_2 = list(chain.from_iterable([set(review) for review in term_freq]))
doc_fd = FreqDist(words_2)  # compute document frequency for each unique word/type
doc_fd


# In[50]:


# [6] 
# [3] w07_act1_gen_feat_vec.ipynb - cell 15
top20_freq_words =doc_fd.most_common(20)
top20_freq_words = [word for word, freq in top20_freq_words]
top20_freq_words


# In[51]:


# [6] 
# removing the top 20 most frequent words
processed_title = term_freq.apply(lambda tokens: [word for word in tokens if word not in top20_freq_words])
processed_title


# In[52]:


# [6]
# statistics on processed titles
def stats_print(tokenized_reviews):
    words = list(chain.from_iterable(tokenized_reviews)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of reviews:", len(tokenized_reviews))
    lens = [len(article) for article in tokenized_reviews]
    print("Average document length:", np.mean(lens))
    print("Maximum document length:", np.max(lens))
    print("Minimum document length:", np.min(lens))
    print("Standard deviation of document length:", np.std(lens))


# In[53]:


# [6]
stats_print(processed_title)


# In[54]:


# [6]
# noticed that minimum document length is 0, hence, computed the number of empty lists
empty_lists = processed_title.apply(lambda x: len(x) == 0).sum()
empty_lists


# In[55]:


# [6]
# appending processed_title column to the original dataframe
clothes_review_data['Processed Title'] = processed_title


# In[56]:


clothes_review_data


# In[57]:


# [6] 
# dropping the rows with empty lists
df_cleaned = clothes_review_data[clothes_review_data['Processed Title'].apply(lambda x: len(x) != 0)]


# In[58]:


# [6] 
# cross verifying if there exists any empty titles in the cleaned dataframe
empty_lists = df_cleaned['Processed Title'].apply(lambda x: len(x) == 0).sum()
empty_lists


# In[59]:


# [6] 
# statistics on processed titles
stats_print(df_cleaned['Processed Title'])


# I reset the index of the DataFrame because, when I drop rows with empty reviews, their corresponding indices are also removed. I encountered an error while splitting the data into train and test becuause the the train_test_split splits on index and there were missing indices. It's good practice to reset the index before exporting the data to maintain readability and avoid potential issues in future tasks. A sequential index ensures clarity and consistency in the DataFrame.

# In[60]:


# [6] 
# resetting the index of the dataframe
df_cleaned = df_cleaned.reset_index(drop=True)


# In[61]:


df_cleaned


# In[62]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 47
count_features_title = cVectorizer.fit_transform([' '.join(title) for title in df_cleaned['Processed Title']]) 


# In[63]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 48
print(count_features_title.shape)


# In[64]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 48
print(count_features_title)


# In[65]:


# [5] Act 4_Document Classification.ipynb - cell 34
# creating training and test split
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(count_features_title, df_cleaned['Recommended IND'], list(range(0,len(df_cleaned))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 500,random_state=seed) # initialising the model
model.fit(X_train, y_train) # training the model
model.score(X_test, y_test)


# In[66]:


labels_title = df_cleaned['Recommended IND']
len(labels_title)


# In[67]:


# [5] Act 4_Document Classification.ipynb - cell 38
cv_df = pd.DataFrame(columns = ['count'],index=range(num_folds)) # creates a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(labels_title)))): # looping through each split

    # extracting labels based on current fold
    y_train = [labels_title[i] for i in train_index]
    y_test = [labels_title[i] for i in test_index]

    # Count Features Model Evaluation
    X_train_count, X_test_count = count_features_title[train_index], count_features_title[test_index]
    cv_df.loc[fold,'count'] = evaluate(count_features_title[train_index],count_features_title[test_index],y_train,y_test,seed)
    
    fold +=1


# In[68]:


# [5] Act 4_Document Classification.ipynb - cell 39
cv_df


# In[69]:


df_cleaned['Combined'] = df_cleaned['Processed Review Text'] + df_cleaned['Processed Title']


# In[70]:


df_cleaned


# In[71]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 47
count_features_combined = cVectorizer.fit_transform([' '.join(title) for title in df_cleaned['Combined']]) 


# In[72]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 48
print(count_features_combined.shape)


# In[73]:


# [3] w07_act1_gen_feat_vec.ipynb - cell 48
print(count_features_combined)


# In[74]:


# [5] Act 4_Document Classification.ipynb - cell 34
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(count_features_combined, df_cleaned['Recommended IND'], list(range(0,len(df_cleaned))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 500,random_state=seed) # initialising the model
model.fit(X_train, y_train) # training the model
model.score(X_test, y_test)


# In[75]:


labels_title = df_cleaned['Recommended IND']
len(labels_title)


# In[76]:


# [5] Act 4_Document Classification.ipynb - cell 38
cv_df = pd.DataFrame(columns = ['count'],index=range(num_folds)) # creating a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(labels_title)))):  # looping through each split
    # extracting labels based on current fold
    y_train = [labels_title[i] for i in train_index]
    y_test = [labels_title[i] for i in test_index]

    # Count Features Model Evaluation
    X_train_count, X_test_count = count_features_combined[train_index], count_features_combined[test_index]
    cv_df.loc[fold,'count'] = evaluate(count_features_combined[train_index],count_features_combined[test_index],y_train,y_test,seed)

    fold +=1


# In[77]:


# [5] Act 4_Document Classification.ipynb - cell 39
cv_df


# ## Q1: Language model comparisons

# The model comparison results indicate that the Count representation consistently delivers better performance than both the Unweighted and Weighted Pretrained Word2Vec representations in classifying clothing reviews. Over five evaluation folds, the Count model achieved the highest accuracy scores, underscoring its effectiveness in capturing relevant features from the review texts. In contrast, the Unweighted and Weighted representations demonstrated similar levels of performance, with the Unweighted model slightly outperforming the Weighted model in certain cases.This indicates that the Count-based model is the most effective for this dataset, with the Weighted and Unweighted representations trailing behind.

# ## Q2: Does more information provide higher accuracy?

# In Task 2, I generated various feature representations for clothing reviews, but I did not take into account other features like the review titles. To determine if adding more information could enhance model accuracy, I performed experiments comparing the performance of classification models based on different input types: the title of the review alone, the review description, and a combination of both. I chose Count Vector Model to make the comparisons.<br>
# 
# The results indicate that using only the title resulted in moderate accuracy scores, with a maximum accuracy of 0.860395 across five folds. In contrast, the description alone achieved higher accuracy, peaking at 0.880153. However, when both the title and detailed review were combined, the accuracy improved further, with the highest accuracy recorded at 0.881152. <br>
# 
# These results indicate that incorporating the title along with the review description improves classification performance, emphasizing the significance of utilizing all available information to boost model accuracy. I still think that if we had more number of tokens in Title column, we could have seen a significant improvement in the accuracy of the model.

# ## Summary
# This part of the assignment primarily focuses on analyzing clothing reviews through feature representation and classification. The steps involved in developing the code helped in understanding the key concepts of word embeddings and vector representation. The activities and lecture slides are designed precisely to handle the required tasks efficiently. The experiments conducted to compare different language models and to explore the impact of incorporating additional information, such as review titles, on model accuracy—using 5-fold cross-validation for robust evaluation—are thought-provoking. This assessment provides a practical opportunity to apply machine learning techniques to natural language processing tasks.

# ## References

# [1] Usage of eval(): https://stackoverflow.com/questions/67323995/how-to-remove-quotes-from-a-list-of-dataframes-in-python <br>
# [2] Canvas/Modules/Week 8 - Activities/w08_activities/w08_act2_embedding_classification.ipynb https://rmit.instructure.com/courses/125024/pages/week-8-activities-2?module_item_id=6449435 <br>
# [3] Canvas/Modules/Week 7 - Activities/w07_activities/w07_act1_gen_feat_vec.ipynb https://rmit.instructure.com/courses/125024/pages/week-7-activities-2?module_item_id=6449422 <br>
# [4] Canvas/Modules/Week 8/Extra Material/Generate Feature Vectors/Act 3_ Generating Feature Vectors.ipynb https://rmit.instructure.com/courses/125024/pages/generate-feature-vectors?module_item_id=6449431 <br>
# [5] Canvas/Modules/Week 8/Extra Material/Activity 4: Document Classification/Act 4_Document Classification.ipynb https://rmit.instructure.com/courses/125024/pages/activity-4-document-classification?module_item_id=6449432 <br>
# [6] task1.ipynb <br>
# 
