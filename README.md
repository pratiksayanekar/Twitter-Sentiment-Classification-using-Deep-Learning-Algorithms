# Twitter-Sentiment-Classification-using-Deep-Learning-Algorithms

### Directory Structure

    .
    ├── checkpoints             # Models with minimum validation loss are saved here
            ├── CNN
            ├── LSTM
            ├── BiLSTM
    ├── Data                    # csv files are stored here
            ├── dataset(clean).csv
            ├── preprocessed_data.csv
            ├── glove.6B.100d.txt
    ├── figs                     # Visualization related to dataset and model evaluations
            ├── Dataset
                ├── Dataset related visualizations
            ├── Models
                ├── Deep Learning model evaluation visualizations
    ├── models                    # Custom python classes with model specific utilities
            ├── BiLSTM.py
            ├── CNN_1.py
            ├── LSTM_1.py
            ├── Word2Vec_1.py
    ├── SavedNotebooks            # as a part of experiment, several other approaches and results are saved here
            ├── GPU_Notebooks/
            ├── Deep_Learning_models_dropout_0.5.ipynb
            ├── Deep_Learning_models_dropout_0.8.ipynb
            ├── Deep_Learning_models_dropout_Glove.ipynb
    ├── utils                     # this file incorporates utilites required for deep neural network modeling
            ├── basic_utilities.py
    ├── Data_Preprocessing.ipynb    # Preprocessing file
    ├── Deep_Learning_Models.ipynb  # Deep Learning model implementation
    ├── requirements.txt            
    └── README.md


# Requirements
> Requirement of this project is generated using below command.
> pip install pipreqs
> pipreqs /path/to/project

There are multiple lines of requirement.txt is generated. So for any version mismatch, requesting to execute below commented cell.

# Specified requirements can be installed with below command.
> pip install -r requirements.txt

# Instructions to execute this project.

Setting up filePath variable: <br>
> Find filePath variable and change the value as per the project location in file system. filePath variable can be found in below files,
- Data_Preprocessing.ipynb <br>
- Deep_Learning_models.ipynb<br>
- /utils/basic_utilities.py<br>
- /models/BiLSTM.py<br>
- /models/CNN_1.py<br>
- /models/LSTM_1.py<br>

# Execute Data_Preprocessing.ipynb file:
> This notebook process the csv file stored at '/Data/dataset(clean).csv'. <br>
> Please make sure that the file present at specified location or change the filePath accordingly. <br>
> It will generate the preprocessed file at '/Data/preprocessed_data.csv'. <br>
> Execute Deep_Learning_models.ipynb: file <br>
> The preprocessed data of previous step is loaded here and further deep neural network modeling is performed. <br>
There are 5 python files created for utility purposes of respective models. These files are imported in import section. <br>
> Section 3 describes the word embedding and there are two approaches discussed there, Word2Vec and GloVe. <br>
> Execute either 3.1 or 3.2 section to perform specified word embeddings. <br>
> glove requires the the source file stored at 'Data/glove.6B.100d.txt'. This file is of 350mb, as per space availability this file will be uploaded. If not present requesting you to store one at specified location. <br>
> For model evaluation purpose, only execute evalution part specified in each model, this will load the previously saved trained model and evaluate on it. <br>
> Model training is done on Macbook pro CPU, so the time taken to execute each batch is tremendously high.<br>

# Project Overview:
title

Evaluation Metric - Accuracy:
Models are tuned on the basis of values provided as filter length, kernal length, max_len and dropout. As per computing power batch size and number of epochs can be changed. Initially the model is trained on batch size of 516 and 1024 as well to analyse the results, these results are stored at SavedNotebooks directory.

CNN Model
title

Accuracy and Loss trend of training and validation data
title

Confusion Matrix of testing data
title

CNN_v2 Model
title

Accuracy and Loss trend of training and validation data
title

Confusion Matrix of testing data
title

LSTM Model
title

Accuracy and Loss trend of training and validation data
title

Confusion Matrix of testing data
title

LSTM_v2 Model
title

Accuracy and Loss trend of training and validation data
title

Confusion Matrix of testing data
title

BiLSTM + CNN Model
title

Accuracy and Loss trend of training and validation data
title

Confusion Matrix of testing data
title

Insights:

The testing accuracy of above model is 90%, the accuracy of angry and disappointed is comparatively less as model is perfectly distinguishing tweets with happy emotion.
The loss of training and validation curve shows no overfitting, provided that the accuracy on unseen data matches the accuracy of validation data.
Model is performing poor, not actually poor but less efficient while predicting between 'angry' and 'disappointed' tweets. The confusion matrix of all models shows similar results.
The execution time of epochs can be reduced by adding several layers of dropout but this is avoided as it results in increase of loss.
The batch size can be increase more than 1024. for example in last saved notebook by increasing batch size to 2048 reduces the number of steps to run one epoch, 587776 samples were trained in 287 steps. On CPU one epoch of BiLSTM takes almost an hour to execute.
Due to Google collab time limit exhaustion, the latest run is executed on Macbook pro CPU with increased batch size.
