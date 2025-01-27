#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Arya Ramesh Patil
# #### Student ID: S4060675
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * numpy
# * nltk
# * chain
# * division
# 
# ## Introduction
# This part of assessment primarily focuses on text pre-processing. While pre-processing also involves stemming & lemmatisation, sentence segmentation, here we focus on tokenisation, case normalisation, removal of stop words and most/less frequent words. Without basic text pre-processing, it is difficult to build a working machine learning model. The activities and lecture slides of week 7 helped me thoroughly to understand the text pre-processing steps in depth and assisted me to develop the code for this particular part of the assignment.
# 

# ## Importing libraries 

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import nltk
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from __future__ import division
from nltk.probability import *


# ### 1.1 Examining and loading data
# On loading the csv file into pandas dataframe, it is observed that the file has 19662 rows and 10 columns. As per the specifications, task 1 involves working on 'Review Text' column. The column 'Review Text' consists of string of words that represent a review on a clothing item. In order to work on this column, I extracted the column in review_text variable to perform pre-processing steps as given.

# In[2]:


# assigning file name to csv_file variable
csv_file = 'assignment3.csv'


# In[3]:


# importing the said csv file into a dataframe
clothes_review_data = pd.read_csv(csv_file, sep = ',')


# In[4]:


# checking the data
clothes_review_data


# In[5]:


# extracting the 'Review Text' column for pre-processing
review_text = clothes_review_data['Review Text']


# In[6]:


# checking the data
review_text


# ### 1.2 Pre-processing data
# The required text pre-processing steps are:
# * Case Normalisation: All the text is converted into lowercase.
# * Tokenisation: Splitting the reviews into tokens.
# * Removing words based on given conditions:
#   - Removing words with length less than 2
#   - Removing stopwords from given stopwords_en.txt file
#   - Removing words based on term frequency
#   - Removing words based on document frequency

# #### Case Normalisation and Tokenisation
# Here I defined a function 'tokenize_reviews' to perform case normalisation and split the reviews into tokens.

# In[7]:


# [1] w07_act1_gen_feat_vec - cell 5
# function for case normalisation and tokensisation
def tokenize_reviews(review_text):
    nl_review = review_text.lower() # converting reviews to lowercase
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?" # regex pattern to tokenise the reviews
    tokenizer = RegexpTokenizer(pattern) # tokeniser to split the reviews based on the regex pattern
    tokenized_review = tokenizer.tokenize(nl_review) # passing normalised reviews to get list of tokens
    return tokenized_review # returning cleaned list of tokens


# In[8]:


tokenized_reviews = review_text.apply(tokenize_reviews) # applying the function to get cleaned list of tokens # [2]
tokenized_reviews # checking the data


# In[9]:


# [1] w07_act1_gen_feat_vec.ipynb - cell 8
# statistics to get an idea of document length, number of tokens, vocabulary size
def stats_print(tokenized_reviews):
    words = list(chain.from_iterable(tokenized_reviews)) # putting tokens in single list
    vocab = set(words) # getting set of unique words
    lexical_diversity = len(vocab)/len(words) # ratio of unique words to total words
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of reviews:", len(tokenized_reviews))
    lens = [len(article) for article in tokenized_reviews] # calculating length of each review
    print("Average document length:", np.mean(lens))
    print("Maximum document length:", np.max(lens))
    print("Minimum document length:", np.min(lens))
    print("Standard deviation of document length:", np.std(lens))


# In[10]:


stats_print(tokenized_reviews)


# In[11]:


stopwords_file = "stopwords_en.txt" #loading the stopwords file


# In[12]:


# [1] w07_act1_gen_feat_vec.ipynb - cell 16
# reading the file 
with open(stopwords_file, 'r') as file: # opens file in read mode
    stopwords = file.read().splitlines() # splits into line such that there is one stopwword per line
stopwords # checking the data


# In[13]:


# getting unique stopwords
stopwords = set(stopwords)
len(stopwords) # checking the number of stopwords


# #### Removing words
# Here I defined a function 'remove_words' to remove words based on the given conditions.

# In[14]:


# function for removal of words
def remove_words(tokens, stopwords):
    tokens = [token for token in tokens if len(token) >= 2] # removing words with the length less than 2
    tokens = [token for token in tokens if token not in stopwords] # removing stopwords 
    return tokens # returning cleaned list of tokens


# In[15]:


# applying the function on tokenized reviews to remove the words on said condition
cleaned_reviews = tokenized_reviews.apply(lambda tokens: remove_words(tokens, stopwords)) # [2]
cleaned_reviews # checking the data


# In[16]:


stats_print(cleaned_reviews) # statistics on cleaned reviews


# In[17]:


# [1] w07_act1_gen_feat_vec.ipynb - cell 11
words = list(chain.from_iterable(cleaned_reviews)) # putting tokens in single list
vocab = set(words) # getting set of unique words


# In[18]:


len(words) # checking the number of tokens


# In[19]:


len(vocab) # checking the number of unique tokens


# In[20]:


# [1] w07_act1_gen_feat_vec.ipynb - cell 12
term_fd = FreqDist(words) # computing term frequency for each unique word


# In[21]:


term_fd # checking the data


# In[22]:


# removing the words that only appear once in the document collection
term_freq = cleaned_reviews.apply(lambda tokens: [word for word in tokens if term_fd[word] > 1]) # [2]


# In[23]:


term_freq # checking the data


# In[24]:


# [1] w07_act1_gen_feat_vec.ipynb - cell 15
words_2 = list(chain.from_iterable([set(review) for review in term_freq])) # putting unique tokens in single list
doc_fd = FreqDist(words_2)  # computing document frequency for each unique word
doc_fd # checking the data


# In[25]:


# [1] w07_act1_gen_feat_vec.ipynb - cell 15
top20_freq_words =doc_fd.most_common(20) # computing the top 20 most common words 
top20_freq_words = [word for word, freq in top20_freq_words] # extracting only the words from doc_fd without their frequencies
top20_freq_words # checking the data


# In[26]:


# removing the top 20 most frequent words
processed_reviews = term_freq.apply(lambda tokens: [word for word in tokens if word not in top20_freq_words]) # [2]


# In[27]:


processed_reviews # checking the data


# In[28]:


len(processed_reviews) # checking the number of reviews


# In[29]:


stats_print(processed_reviews) # statistics on processed reviews


# In[30]:


# noticed that minimum document length is 0, hence, computed the number of empty lists
empty_lists = processed_reviews.apply(lambda x: len(x) == 0).sum() # [2]
empty_lists


# In[31]:


# appending processed_reviews column to the original dataframe
clothes_review_data['Processed Review Text'] = processed_reviews


# In[32]:


clothes_review_data # checking the data


# I am dropping reviews that result in empty lists because techniques for handling empty instances have not been covered yet. Additionally, there is no suitable way to replace empty reviews with meaningful content, as the data type is text. Furthermore, we were instructed to drop empty instances during the lectorial session.

# In[33]:


# dropping the rows with empty lists
df_cleaned = clothes_review_data[clothes_review_data['Processed Review Text'].apply(lambda x: len(x) != 0)] # [2]
df_cleaned # checking the data


# In[34]:


# cross verifying if there exists any empty reviews in the cleaned dataframe
empty_lists = df_cleaned['Processed Review Text'].apply(lambda x: len(x) == 0).sum() # [2]
empty_lists


# In[35]:


stats_print(df_cleaned['Processed Review Text']) # statistics on processed reviews


# I reset the index of the DataFrame because, when I drop rows with empty reviews, their corresponding indices are also removed. It's good practice to reset the index before exporting the data to maintain readability and avoid potential issues in future tasks. A sequential index ensures clarity and consistency in the DataFrame.

# In[36]:


# resetting the index of the dataframe
df_cleaned = df_cleaned.reset_index(drop=True) # [3]


# ## Saving required outputs
# Saving the requested information as per specification.
# - processed.csv
# - vocab.txt

# In[37]:


# saving the processed data 'processed.csv' file
df_cleaned.to_csv('processed.csv', index=False)


# In[38]:


# combining list of reviews into a single list and then getting only unique set of words and ordering the reviews alphabetically
vocabulary = sorted(set(chain.from_iterable(processed_reviews))) # [4]
vocabulary # checking the data


# In[39]:


# [1] w07_act1_gen_feat_vec.ipynb - cell 55
out_file = open("./vocab.txt", 'w') # creating a file and opening it in write mode

# looping through each word in the vocabulary using its index 
for ind in range(0, len(vocabulary)):
    out_file.write(f"{vocabulary[ind]}:{ind}\n") # writing to a file in 'word_string:word_integer_index' format
                   
out_file.close() # closing the file


# ## Summary
# This part of the assignment highlights the importance of text pre-processing and its vital role in building machine learning models. This task is fundamental for preparing text data for further analysis, such as creating document vectors and feeding these vector representations into machine learning models for classification. It is crucial to follow the exact formatting requirements, especially for the vocabulary file, to ensure compatibility with subsequent tasks. In addition, the activities and lectorial material are designed perfectly to carry out these tasks and helped me understand the concepts thoroughly.

# ## References

# [1] Canvas/Modules/Week 7 - Activities/w07_activities/w07_act1_gen_feat_vec.ipynb https://rmit.instructure.com/courses/125024/pages/week-7-activities-2?module_item_id=6449422 <br>
# [2] Usage of apply(): https://stackoverflow.com/questions/36213383/pandas-dataframe-how-to-apply-function-to-a-specific-column <br>
# [3] Usage of reset_index(): https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-dataframe <br>
# [4] Usage of sorted(): https://stackoverflow.com/questions/32072076/find-the-unique-values-in-a-column-and-then-sort-them <br>
