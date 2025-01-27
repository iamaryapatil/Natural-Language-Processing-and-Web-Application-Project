# Natural-Language-Processing-and-Web-Application-Project
Developed an end-to-end pipeline for text preprocessing, feature extraction, machine learning, and web-based application development, focusing on clothing review classification and recommendation. The project involved parsing and cleansing raw data, building and evaluating machine learning models, and deploying the models in an interactive Flask-based online shopping platform.

Natural Language Processing (Milestone I)

- Text Preprocessing: Implemented a comprehensive pipeline for tokenization, normalization, stopword removal, and handling infrequent and overused terms in "Review Text."
- Feature Engineering:
  - Created unweighted and TF-IDF-weighted feature representations for textual data using pre-trained GloVe embeddings.
  - Constructed count vector representations for word frequency analysis.
- Machine Learning:
  - Built and optimized multiple classification models (e.g., logistic regression) to predict recommendation labels for clothing items.
  - Conducted 5-fold cross-validation to evaluate models and analyze performance metrics like accuracy, precision, and recall.

Web-based Application Development (Milestone II)

- Flask Application: Designed and deployed an online shopping website using Flask.
  - Allowed users to browse clothing items and their reviews.
  - Enabled shoppers to create new reviews that were programmatically assigned recommendation labels (recommended or not recommended) using the      trained machine learning model from Milestone I.
- Dynamic Features:
  - Implemented a robust search functionality to retrieve unique clothing items based on keywords, handling word variations (e.g., “dress” and       “dresses”).
  - Integrated review management, enabling users to view, add, and update reviews in real-time.
- Data Persistence: Ensured newly added reviews were appended to a CSV file, maintaining data consistency and usability for future analysis.
