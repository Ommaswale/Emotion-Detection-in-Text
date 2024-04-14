# Emotion-Detection-in-Text
This Jupyter Notebook implements a machine learning pipeline to detect emotions from textual data. It leverages natural language processing (NLP) techniques for text cleaning and feature engineering, followed by logistic regression for emotion classification.
Key Steps:

1. Data Loading and Exploration:

Imports necessary libraries like pandas, NumPy, seaborn, neattext, scikit-learn.
Loads the emotion dataset (emotion_dataset_raw.csv).
Explores the distribution of emotions using value counts and visualizations (Seaborn countplot).

2. Data Cleaning:

Utilizes neattext's remove_userhandles function to eliminate user mentions (e.g., "@username").
Applies remove_stopwords function to remove common, non-meaningful words from the text.

3. Feature Engineering and Labels:

Identifies features: Cleaned text (Clean_Text).
Defines labels: Emotion categories (Emotion).

4. Data Splitting:

Splits the data into training and testing sets using train_test_split (70% for training, 30% for testing).
Sets a random state (42) for reproducibility.

5. Model Building with Pipeline:

Employs Pipeline from scikit-learn to streamline model creation.
Constructs a pipeline with two stages:
CountVectorizer: Converts text data into numerical features (term frequencies).
LogisticRegression: Classifies text into different emotion categories.

6. Training and Evaluation:

Trains the pipeline on the training data (x_train, y_train).
Evaluates performance on the testing data (x_test, y_test).
Calculates accuracy (pipe_lr.score) and prints the result.

7. Prediction and Probability:

Demonstrates making a prediction for a new text sample (ex1).
Shows how to obtain prediction probabilities using pipe_lr.predict_proba.

8. Confusion Matrix and Classification Report:

Generates a confusion matrix to visualize model performance on different emotion categories.
Creates a classification report that provides detailed metrics like precision, recall, F1-score, and support for each emotion class.

