import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import scikitplot as skplt

import sys
filePath = '/Users/pratiksayanekar/Documents/DL_20200161'
#filePath = '/content/drive/My Drive/DeepLearning'
sys.path.append(filePath)

class Utils:

    def __init__(self):
        print("Initialization of Utilities...")

    def split_data(self, data):
        return train_test_split(data.text, data.emotion, test_size=0.20, shuffle=True,
                                stratify=data.emotion, random_state=0)

    def max_len(self, data):
        max_len = 0
        for i in data['text']:
            split_i = i.split()
            if len(split_i) > max_len:
                max_len = len(split_i)
        return max_len

    def tokenization_padding(self, train_data, test_data, max_len):
        max_fatures = 200000  # the number of words to be used for the input of embedding layer

        tokenizer = Tokenizer(num_words=max_fatures, split=' ')  # Create the instance of Tokenizer
        tokenizer.fit_on_texts(train_data.values)

        train_converted = tokenizer.texts_to_sequences(train_data.values)
        test_converted = tokenizer.texts_to_sequences(test_data.values)

        train_converted = pad_sequences(train_converted,
                                        maxlen=max_len)  # Turning the vectors of train data into sequences
        test_converted = pad_sequences(test_converted,
                                       maxlen=max_len)  # Turning the vectors of test data into sequences

        return train_converted, test_converted

    def tokenization_padding_oov_token(self, train_data, test_data, max_len):
        max_fatures = 200000  # the number of words to be used for the input of embedding layer

        tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
        tokenizer.fit_on_texts(train_data.values)

        train_converted = tokenizer.texts_to_sequences(train_data.values)
        test_converted = tokenizer.texts_to_sequences(test_data.values)

        train_converted = pad_sequences(train_converted,
                                        maxlen=max_len)  # Turning the vectors of train data into sequences
        test_converted = pad_sequences(test_converted,
                                       maxlen=max_len)  # Turning the vectors of test data into sequences

        return train_converted, test_converted, tokenizer

    def one_hot_target_variable(self, train_target, test_target):
        target_converted_train = pd.get_dummies(train_target).values  # One-hot expression
        target_converted_test = pd.get_dummies(test_target).values
        return target_converted_train, target_converted_test

    def plot_accuracy_loss(self, history, model):
        acc, val_acc = history.history['acc'], history.history['val_acc']
        loss, val_loss = history.history['loss'], history.history['val_loss']
        epochs = range(len(acc))

        fig = plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        fig.savefig('{}/figs/Models/{}_acc_loss.png'.format(filePath, model))

    def plot_confusion_matrix(self, test, pred, model):
        test = np.argmax(test, axis=1)
        pred = np.argmax(pred, axis=1)
        matrix = confusion_matrix(test, pred)
        print("Confusion Matrix: \n", matrix)
        fig = plt.figure(figsize=(12, 6))
        sns.heatmap(matrix.astype('float') / np.sum(matrix, axis=1), annot=True,
                    fmt='.2%', cmap='Blues', xticklabels=['Angry', 'Disappointed', 'Happy'],
                    yticklabels=['Angry', 'Disappointed', 'Happy'])
        plt.title("Confusion Matrix")
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()
        fig.savefig('{}/figs/Models/{}_confusion_matrix.png'.format(filePath, model))
        return test, pred

    def plot_confusion_matrix_1(self, test, pred):
        test = np.argmax(test, axis=1)
        pred = np.argmax(pred, axis=1)
        skplt.metrics.plot_confusion_matrix(
            test,
            pred,
            figsize=(12, 6),
            title="Confusion Matrix")
