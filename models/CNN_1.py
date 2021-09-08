from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import sys
filePath = '/Users/pratiksayanekar/Documents/DL_20200161'
#filePath = '/content/drive/My Drive/DeepLearning'
sys.path.append(filePath)

class CNN_Model:
    def __init__(self):
        self.callbacks = [ReduceLROnPlateau(monitor='val_loss',patience=5,cooldown=0),
                          EarlyStopping(monitor='val_loss',patience=5, min_delta=0.0001),
                          ModelCheckpoint(filepath='{}/checkpoints/CNN/'.format(filePath),
                                          monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
        self.callbacks_v2 = [ReduceLROnPlateau(monitor='val_loss',patience=5,cooldown=0),
                             EarlyStopping(monitor='val_loss',patience=5,min_delta=0.0001),
                             ModelCheckpoint(filepath='{}/checkpoints/CNN_v2/'.format(filePath),
                                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
        print("CNN Model Initialization...")

    def create_model(self, embed_dim, num_filters, input_len, feature, drop_out, num_words):
        model = Sequential()
        model.add(Embedding(feature, embed_dim, input_length=input_len))
        model.add(SpatialDropout1D(drop_out))
        model.add(Conv1D(filters=num_filters, kernel_size=num_words, activation="relu"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["acc"])
        print("Model Created Successfully...")
        return model

    def create_model_v2(self, embed_dim, num_filters, input_len, feature, drop_out, num_words):
        model = Sequential()
        model.add(Embedding(feature, embed_dim, input_length=input_len))
        model.add(SpatialDropout1D(drop_out))
        model.add(Conv1D(filters=num_filters, kernel_size=num_words, activation="relu", padding = "same"))
        model.add(Conv1D(filters=num_filters, kernel_size=num_words, activation="relu", padding = "same"))
        model.add(SpatialDropout1D(drop_out))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["acc"])
        print("Model Created Successfully...")
        return model
    
    # This method takes one additional parameter as embedding_matrix as weights to embedding layer
    def create_model(self, embed_dim, num_filters, input_len, feature, drop_out, num_words, embedding_matrix):
        model = Sequential()
        model.add(Embedding(feature, embed_dim, input_length=input_len, weights=[embedding_matrix], trainable=True))
        model.add(SpatialDropout1D(drop_out))
        model.add(Conv1D(filters=num_filters, kernel_size=num_words, activation="relu"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["acc"])
        print("Model Created Successfully...")
        return model

    # This method takes one additional parameter as embedding_matrix as weights to embedding layer
    def create_model_v2(self, embed_dim, num_filters, input_len, feature, drop_out, num_words, embedding_matrix):
        model = Sequential()
        model.add(Embedding(feature, embed_dim, input_length=input_len, weights=[embedding_matrix], trainable=True))
        model.add(SpatialDropout1D(drop_out))
        model.add(Conv1D(filters=num_filters, kernel_size=num_words, activation="relu", padding = "same"))
        model.add(Conv1D(filters=num_filters, kernel_size=num_words, activation="relu", padding = "same"))
        model.add(SpatialDropout1D(drop_out))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["acc"])
        print("Model Created Successfully...")
        return model

    def load_saved_model(self):
        return models.load_model('{}/checkpoints/CNN/'.format(filePath))

    def load_saved_model_v2(self):
        return models.load_model('{}/checkpoints/CNN_v2/'.format(filePath))