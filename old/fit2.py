import pandas as pd
import string
import re
from typing import List

class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        # Dictionaries of probabilities of words for two labels
        self.probs = {"neutral": dict(), "discrim": dict()}

        # Count of words for two labels
        self.words = {"neutral": 0, "discrim": 0}

        # Count of tweets for two labels
        self.tweets = {"neutral": 0, "discrim": 0}

        # Count of unique words & alpha param
        self.unique_words = 0
        self.a = 1

    def fit(self, tweets: List[str], labels: List[str]):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param x: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """
        def tokenize(lst):
            for index in range(len(lst)):
                lst[index] = [item for item in lst[index].split(" ") if item != '']
            print(lst)

        def add_tweet_to_dict(tweet: List[str], label: str):
            other_label = "discrim" if label == "neutral" else "neutral"

            for word in tweet:
                if word in self.probs[label].keys():
                    self.probs[label][word] += 1
                    self.words[label] += 1
                else:
                    self.probs[label][word] = 1
                    self.words[label] += 1
                    self.unique_words += 1
                
                if word not in self.probs[other_label].keys():
                    self.probs[other_label][word] = 0

        def convert_frequency_to_probability(label: str, a: int) -> None:
            words_with_a = self.words[label] + a * self.unique_words

            for word, count in self.probs[label].items():
                self.probs[label][word] = (count + a) / words_with_a

        tokenize(tweets)

        for tweet, label in zip(tweets, labels):
            self.tweets[label] += 1
            add_tweet_to_dict(tweet, label)

        print()

        # print(self.probs[label]["@user"])

        convert_frequency_to_probability("neutral", self.a)
        convert_frequency_to_probability("discrim", self.a)

    def predict_prob(self, tweet: str, label: str) -> float:
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param tweet: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        # Set initial probability to the P(label)
        probability = self.tweets[label] / (self.tweets["neutral"] + self.tweets["discrim"])

        # Multiply by P(word|label)
        for word in tweet:
            if word in self.probs[label]:
                probability *= self.probs[label][word]

        return probability

    def predict(self, tweet: str) -> str:
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
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

        # in this score, classifying 'neutral' is as important as 'discrim'.
        accuracy = (discrim_corr + neutral_corr) / (discrim_all + neutral_all)# 0.5 * discrim_corr / discrim_all + 0.5 * neutral_corr / neutral_all 
        return round(accuracy * 100, 2), discrim_corr, discrim_all, neutral_corr, neutral_all

    def __str__(self):
        return f"Word count: {self.words}\nTweets: {self.tweets}\nUnique words: {self.unique_words}\n"


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

    # print(f"{train_X = }")
    # print(len(test_X))
    # print(len(test_y))


    # print(f"{train_y = }")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    classifier.predict_prob(test_X[0], test_y[0])

    print("--"*10)
    print(f"model score: {classifier.score(test_X, test_y)}%")
    print("--"*10)