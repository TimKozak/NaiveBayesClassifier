from random import random
import pandas as pd
import string
import re

class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        # Dictionaries of probabilities of words for two labels
        self.prob_word_if_label = {"neutral": dict(), "discrim": dict()}

        # Count of words for both labels
        self.label_words_count = {"neutral": 0, "discrim": 0}

        # Count of tweets for both labels
        self.labels_count = {"neutral": 0, "discrim": 0}

        # Count of unique words & alpha param
        self.unique_words = set()
        self.alpha = 1

    def fit(self, tweets: list, labels: list) -> None:
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param tweets: pd.DataFrame|list - train input/messages
        :param labels: pd.DataFrame|list - train output/labels
        :return: None
        """
        def tokenize(lst):
            for index in range(len(lst)):
                lst[index] = [item for item in lst[index].split(" ") if item != '']

        def add_tweet_to_dict(tweet, label) -> None:
            """Fill in dictionaries with all words"""
            other_label = "discrim" if label == "neutral" else "neutral"

            for word in tweet:
                self.unique_words.add(word)
                if word in self.prob_word_if_label[label]:
                    self.prob_word_if_label[label][word] += 1
                else:
                    self.prob_word_if_label[label][word] = 1
                
                self.label_words_count[label] += 1
                
                if word not in self.prob_word_if_label[other_label]:
                    self.prob_word_if_label[other_label][word] = 0

        def convert_frequency_to_probability(label: str) -> None: 
            """Convert frequency to probability with param alpha for handling 0 probabilities"""
            words_total_a = self.label_words_count[label] + self.alpha * len(self.unique_words)

            for word, count in self.prob_word_if_label[label].items():
                # Compute P(word|label)
                self.prob_word_if_label[label][word] = (count + self.alpha) / words_total_a
        
        tokenize(tweets)

        # Iterate through all tweets and make dictionaries of word frequencies
        for tweet, label in zip(tweets, labels):
            self.labels_count[label] += 1
            add_tweet_to_dict(tweet, label)

        # Convert frequency dictionaries to probability dictionaries
        convert_frequency_to_probability("neutral")
        convert_frequency_to_probability("discrim")

    def predict_prob(self, tweet: str, label: str) -> float:
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param tweet: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        # Set initial probability to the P(label)
        probability = self.labels_count[label] / (self.labels_count["neutral"] + self.labels_count["discrim"])

        other_label = "discrim" if label == "neutral" else "neutral"

        # Multiply by P(word|label)
        for word in tweet.split():
            if word in self.prob_word_if_label[label]:
                probability *= self.prob_word_if_label[label][word] / (
                    self.prob_word_if_label[label][word] + self.prob_word_if_label[other_label][word]
                )  # P(feature|class) / P(feature)

        return probability

    def predict(self, tweet: str) -> str:
        """
        Predict label for a given tweet.
        :param tweet: str - tweet
        :return: str - label that is most likely to be truly assigned to a given tweet
        """
        # Calculate probability for both labels
        prob_discrim = self.predict_prob(tweet, "discrim")
        prob_neutral = self.predict_prob(tweet, "neutral")

        # Assign label too whichever probability is bigger
        label = "discrim" if prob_discrim > prob_neutral else "neutral"
        return label

    def score(self, X: list, y: list) -> float:
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        :return:
        """
        discrim_all = discrim_corr = neutral_all = neutral_corr = 0

        for tweet, correct_label in zip(X, y):
            if correct_label == "discrim":
                discrim_corr += correct_label == self.predict(tweet)
                discrim_all += 1
            else:
                neutral_corr += correct_label == self.predict(tweet)
                neutral_all += correct_label == "neutral"

        accuracy = (discrim_corr + neutral_corr) / (discrim_all + neutral_all)
        return round(accuracy * 100, 2) # , discrim_corr, discrim_all, neutral_corr, neutral_all 

    def __str__(self):
        return f"Word count: {self.label_words_count}\nTweets: {self.labels_count}\nUnique words: {len(self.unique_words)}\n"

def process_data(data_file: str) -> tuple:
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """

    df = pd.read_csv(data_file)

    with open("./data/stop_words.txt", mode="r", encoding="ascii") as stop_words_file:
        stop_words = [word.strip("\n") for word in stop_words_file.readlines()]

    punctuation = string.punctuation.replace("@", "")
    redundantchars = "â¦"
    punctuation_trans = str.maketrans(punctuation, " " * len(punctuation), redundantchars)
    remove_nonascii_re = re.compile(r"(&amp;|[^\x00-\x7F]+)")

    def filter_tweets(tweet: str) -> str:
        """Filter a tweet from puntuation and stopwords"""
        tweet = " " + re.sub(remove_nonascii_re, '', tweet).lower().translate(punctuation_trans) + " "
        for word in stop_words:
            tweet = tweet.replace(" " + word + " ", " ")
        tweet = tweet.strip(" ")
        return tweet

    df["tweet"] = df["tweet"].apply(filter_tweets)

    tweets, labels = df["tweet"].values, df["label"].values
    return tweets, labels


if __name__ == "__main__":
    train_X, train_y = process_data("./data/train.csv")
    test_X, test_y = process_data("./data/test.csv")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    classifier.predict_prob(test_X[0], test_y[0])

    print("--"*10)
    print(f"model score: {classifier.score(test_X, test_y)}%")
    print("--"*10)