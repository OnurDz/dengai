import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.feature_selection import SelectKBest as kbest
from sklearn.feature_selection import mutual_info_regression as mir
from sklearn.feature_selection import f_regression as freg



def res(city, ep):

    dataset = pd.read_csv('../data/dengue_features_train.csv', index_col=[0, 1, 2])
    dataset.drop('week_start_date', axis=1, inplace=True)
    labels = pd.read_csv('../data/dengue_labels_train.csv', index_col=[0, 1, 2])
    labels = labels.shift(periods=1, fill_value=0)
    dataset = dataset.loc[city]
    labels = labels.loc[city]
    dataset.fillna(method='ffill', inplace=True)

    cordat = dataset.copy()
    cordat['total_cases'] = labels['total_cases']
    cor = cordat.corr()

    cor_target = abs(cor["total_cases"])
    selected_features = cor_target[cor_target >= 0.07]
    selected_features.drop('total_cases', inplace=True)
    #print(selected_features)
    dataset = dataset[selected_features.keys()]
    print(dataset.shape)
    #exit()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    train_labels = labels.sample(frac=0.8, random_state=0)

    test_dataset = dataset.drop(train_dataset.index)
    test_labels = labels.drop(train_dataset.index)

    def norm(x):
      scaler = MinMaxScaler()
      return scaler.fit_transform(x)
      #return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    def build_model():
      model = keras.Sequential([
        layers.Dense(13, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(9, activation='relu'),
        layers.Dense(1)
      ])

      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    
      model.compile(loss='mae',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
      return model

    model = build_model()

    EPOCHS = ep

    history = model.fit(
      normed_train_data, train_labels,
      epochs=EPOCHS, validation_split = 0.33, verbose=0,
      shuffle=False, callbacks=[tfdocs.modeling.EpochDots()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "loss")
    #plt.ylim([0, 10])
    plt.ylabel('MAE [total_cases]')
    plt.savefig('iqmaelarge.png')
    plt.show()



    test_predictions = model.predict(normed_test_data).flatten().round().astype(np.int)
    test_predictions = np.nan_to_num(test_predictions)
    loss = mae(test_labels, test_predictions)
    #print(test_predictions)
    print('MEAN ABSOLUTE ERROR:', loss)
    

    result = []
    for i in test_predictions:
        result.append(round(i))

    return loss

def main():
    #loss = []
    #eps = []
    #ep = 50
    #for i in range(50):
    #  print('Iteration', i)
    #  loss.append(res('sj', ep))
    #  eps.append(ep)
    #  ep = ep + 50
    
    #plt.plot(eps, loss)
    #plt.xlabel('EPOCH')
    #plt.ylabel('Mean Absolute Error')
    #plt.savefig('sjepoch50.png')
    #plt.show()

    sj = res('sj', 2500)
    #iq = res('iq')
    #result = sj + iq
    #print(len(result))
    #df = pd.read_csv("../data/submission_format.csv", index_col=[0, 1, 2])
    #df['total_cases'] = result
    #df.to_csv('sub.csv')

if __name__ == '__main__':
    main()