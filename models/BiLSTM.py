from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding, SpatialDropout1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import sys
filePath = '/Users/pratiksayanekar/Documents/DL_20200161'
#filePath = '/content/drive/My Drive/DeepLearning'
sys.path.append(filePath)

class BiLSTM_Model:
    def __init__(self):
        self.callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                          EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5),
                          ModelCheckpoint(filepath='{}/checkpoints/BiLSTM/'.format(filePath),
                                          monitor='val_loss', save_best_only=True, mode='min')]
        print("BiLSTM Model Initialization...")

    def create_model(self, embed_dim, lstm_out, input_len, feature, drop_out, embedding_matrix):
        embedding_layer = Embedding(input_dim=feature,
                                    output_dim=embed_dim,
                                    weights=[embedding_matrix],
                                    input_length=input_len,
                                    trainable=True)

        model = Sequential()
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(lstm_out, dropout=drop_out, return_sequences=True)))
        model.add(SpatialDropout1D(drop_out))
        model.add(Bidirectional(LSTM(lstm_out, dropout=drop_out, return_sequences=True)))
        model.add(Conv1D(lstm_out, 5, activation='relu'))
        model.add(GlobalMaxPool1D())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        print("Model Created Successfully...")
        return model

    def load_saved_model(self):
        return models.load_model('{}/checkpoints/BiLSTM/'.format(filePath))
