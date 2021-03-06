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

        # Set of unique words & alpha param
        self.unique_words = set()
        self.alpha = 0.154

    def tokenize(self, lst):
        new_tweets = []
        for index in range(len(lst)):
            sp = lst[index].split()  # grams
            bigrams = set(map(lambda x: x[0] + x[1], zip(sp[:-1], sp[1:])))
            new_tweets.append(set(sp) | bigrams)  # dict(map(lambda word: (word, sp.count(word)), set(sp)))
        return new_tweets

    def fit(self, tweets: list, labels: list) -> None:
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param tweets: pd.DataFrame|list - train input/messages
        :param labels: pd.DataFrame|list - train output/labels
        :return: None
        """
        def add_tweet_to_dict(tweet, label) -> None:
            """Fill in dictionaries with all words"""
            other_label = "discrim" if label == "neutral" else "neutral"

            for word in tweet:
                if word in self.prob_word_if_label[label]:
                    self.prob_word_if_label[label][word] += 1
                else:
                    self.prob_word_if_label[label][word] = 1

                self.label_words_count[label] += 1
                self.unique_words.add(word)

                if word not in self.prob_word_if_label[other_label]:
                    self.prob_word_if_label[other_label][word] = 0

        def convert_frequency_to_probability(label: str) -> None: 
            """Convert frequency to probability with param alpha for handling 0 probabilities"""
            # Laplace smoothing (remove P(word|label)=0)
            words_total_a = self.label_words_count[label] + self.alpha * len(self.unique_words)

            for word, count in self.prob_word_if_label[label].items():
                # Compute P(word|label)
                self.prob_word_if_label[label][word] = (count + self.alpha) / words_total_a

        tweets = self.tokenize(tweets)

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

        # Multiply by P(word|label) / P(word)
        for word in self.tokenize([tweet])[0]:
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
        discrim_false = discrim_true = neutral_false = neutral_true = 0

        for tweet, correct_label in zip(X, y):
            if correct_label == "discrim":
                discrim_true += correct_label == self.predict(tweet)
                discrim_false += correct_label != self.predict(tweet)
            else:
                neutral_true += correct_label == self.predict(tweet)
                neutral_false += correct_label != self.predict(tweet)

        # F-Score for correctly predicting discrimination tweets
        accuracy = discrim_true / (discrim_true + 0.5 * (discrim_false + neutral_false))
        oldacc = (discrim_true + neutral_true) / (discrim_false + neutral_false + discrim_true + neutral_true)
        return round(accuracy * 100, 2), round(oldacc * 100, 2)

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

    # regexp to remove non-acsii
    remove_nonascii_re = re.compile(r"(&amp;|[^\x00-\x7F]+)")
    # punctuation to replace by spaces
    punctuation_sp = "!\"#$%()*,-./:;?@[\]{|}~"
    # punctuation to remove
    punctuation_rm = string.punctuation.translate(str.maketrans("", "", punctuation_sp))
    # make the translation
    punctuation_trans = str.maketrans(punctuation_sp, " " * len(punctuation_sp), punctuation_rm)

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