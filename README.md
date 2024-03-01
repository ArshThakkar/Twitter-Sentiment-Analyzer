# Twitter Sentiment Analyzer using Machine Learning

This project aims to analyze the sentiment of tweets using machine learning techniques. It utilizes a dataset from Kaggle, performs data preprocessing, trains a logistic regression model, and evaluates its performance. The trained model can be used to predict the sentiment of new tweets.

## Installation

To run this project, follow these steps:

1. Install the Kaggle library:

```bash
pip install kaggle
```

2. Upload your `kaggle.json` file and configure its path:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. Import the Twitter sentiment dataset from Kaggle:

```bash
kaggle datasets download -d kazanova/sentiment140
```

4. Extract the compressed dataset.

## Data Processing

1. Load the dataset into a pandas DataFrame.
2. Rename columns and handle missing values.
3. Convert target labels from 4 to 1 (positive sentiment) for consistency.
4. Perform text preprocessing, including stemming and removing stopwords.

## Model Training

1. Split the dataset into training and test sets.
2. Convert text data into numeric data using TF-IDF vectorization.
3. Train a logistic regression model on the training data.
4. Evaluate the model's accuracy on both training and test data.

## Model Evaluation

The logistic regression model achieves an accuracy of approximately 77.8% on the test data.

## Saving and Using the Model

1. Save the trained model using pickle.
2. Load the saved model for future predictions.

## Usage

After setting up the environment and training the model, you can use it to predict the sentiment of new tweets. Simply provide the text of the tweet to the model, and it will output whether the tweet is positive or negative.

```python
import pickle

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Example usage
X_new_tweet = "I love this project!"
prediction = loaded_model.predict(X_new_tweet)

if prediction[0] == 0:
    print('Negative Tweet')
else:
    print('Positive Tweet')
```

## Dependencies

- numpy
- pandas
- scikit-learn
- nltk

Make sure to install these dependencies before running the project.

## Contributors

- Arsh Thakkar

Feel free to contribute to this project by forking and submitting pull requests!
