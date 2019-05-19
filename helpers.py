import pandas as pd
import numpy as np 
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pylab



def load_and_preprocess(file_path):
    #load the file
    data = pd.read_csv(file_path,low_memory=False, parse_dates=True, header = None)
   
    data.replace('?', np.NaN, inplace = True)
    #replace object values into floats
    data.iloc[:,1:] = data.iloc[:,1:].apply(lambda x : pd.to_numeric(x,errors='coerce'))
    #replace missing values by the mean
    data.fillna(data[data[73]==0].mean(axis = 0),axis = 0, inplace = True)
    #drop the date column
    train_data = data.drop(0,axis = 1)
    return data, train_data


#Method to prepare data for lstm autoencoder taken from 'https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/' and modified to only keep what is interesting

# split a multivariate sequence into samples of step n_steps to fit an lstm
def split_sequences(sequences, n_steps):
    """ split the sequences into batchs of subsequences using
        the n_steps parameter """
        
    X =  list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :-1]
        X.append(seq_x)

    return np.array(X)

#standerdize the data and put it in the right format
def prepare_training_data(data,n_steps,scaler):
    """ scale the data using a standerscaler and split the sequences """
    data_stand = scaler.transform(data)
    X = split_sequences(data_stand,n_steps)
    return X

# methods to fit the model and save it 
def fit_model(autoencoder, X_train, y_train, X_val, nb_epoch, path):
    """ fit the model and saves the model logs """
    
    cp = ModelCheckpoint(filepath=path,
                                   save_best_only=True,
                                   verbose=0)
    tb = TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)
    history = autoencoder.fit(X_train,y_train,
                        epochs=nb_epoch,
                        validation_data = (X_val,X_val),
                        verbose=0,
                        callbacks=[cp, tb]).history
    return history

#another method to add noise to the data
def add_noise_to_data(data,noise_factor,replace_factor):
    """ add noise to the data by replacing random values by 0 """
    xs = np.random.randint(0, high=data.shape[0], size=int(noise_factor *data.shape[0]))
    ys = np.random.randint(0, high=data.shape[1], size=int(noise_factor *data.shape[0]))
    for i,j in zip(xs,ys):
        data.iloc[i,j] = replace_factor*data.iloc[i,j]
    return data

def plot_model_behavior(history,metric):
    """ Plot the metric vs epoch for the train and validation set"""
    plt.plot(history[metric])
    plt.plot(history['val_' +metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('nb_epoch')
    plt.legend(['train', 'validation'], loc='center right')
    


def plot_graph(history, metric1='acc',metric2='cosine_proximity', metric3='loss'):
    """ plot the loss, the accuracy and cosine proximity """

    plt.figure(figsize=(15,5))

    gs = gridspec.GridSpec(1,3)

    ax = plt.subplot(gs[0])
    plot_model_behavior(history,metric1)

    ax = plt.subplot(gs[1])
    plot_model_behavior(history,metric2)

    ax = plt.subplot(gs[2])
    plot_model_behavior(history,metric3)

    
def compute_mse(X_test,predictions):
    """compute the mean square error."""
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[2]))
    predictions =  predictions.reshape((predictions.shape[0],predictions.shape[2]))
    return np.mean((X_test - predictions)**2,axis=1)


def get_predictions(X, model, threshold):
    """ Get the 0/1 value for each input using the threshold on the reconstruction score """
    mse = compute_mse(X,model.predict(X))
    y_pred = [1 if error>=threshold else 0 for error in mse]
    return np.array(y_pred)


def plot_ROC(y_test, x_test, model):
    """ Plot the ROC curve to compare the two models: baseline and pretrained"""
    
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_test, compute_mse(x_test,model.predict(x_test)))
    roc_auc = auc(false_pos_rate, true_pos_rate,)

    plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
    plt.plot([0,1],[0,1], linewidth=5)

    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def confusion_matrix(y_test,y_pred):
    """ plot the confusion matrix """
    plt.figure(figsize=(10, 5))
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    
def plot_recall_precision(precision, recall, thresholds):
    """ plot recall precision curve to choose the threshold """
    plt.figure(figsize=(10,5))
    plt.plot(thresholds, precision[:-1], label="Precision",linewidth=1)
    plt.plot(thresholds, recall[:-1], label="Recall",linewidth=1)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.xscale('log')
    plt.legend(loc='center right')
    plt.show()



