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


def res(city):
    train_dataset = pd.read_csv('../data/dengue_features_train.csv', index_col=[0, 1, 2])
    train_dataset = train_dataset.loc[city]
    test_dataset = pd.read_csv('../data/dengue_features_test.csv', index_col=[0, 1, 2])
    test_dataset = test_dataset.loc[city]

    train_dataset.drop('week_start_date', axis=1, inplace=True)
    train_dataset.fillna(method='ffill', inplace=True)
    test_dataset.drop('week_start_date', axis=1, inplace=True)

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    #print(train_stats)

    train_labels = pd.read_csv('../data/dengue_labels_train.csv', index_col=[0, 1, 2])
    train_labels = train_labels.loc[city]
    train_labels.shift(periods=2, fill_value=train_labels['total_cases'].mean())

    cordat = train_dataset.copy()
    cordat['total_cases'] = train_labels['total_cases']
    cor = cordat.corr()
    #plt.figure(figsize=(12,10))
    #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()

    cor_target = abs(cor["total_cases"])
    selected_features = cor_target[cor_target >= 0.07]
    selected_features.drop('total_cases', inplace=True)
    #print(selected_features)
    train_dataset = train_dataset[selected_features.keys()]
    test_dataset = test_dataset[selected_features.keys()]


    def norm(x):
      scaler = MinMaxScaler()
      return scaler.fit_transform(x)

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    def build_model():
      model = keras.Sequential([
        layers.Dense(13, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(9, activation='relu'),
        layers.Dense(1)
      ])

      optimizer = tf.keras.optimizers.RMSprop(0.001)
    
      model.compile(loss='mae',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
      return model

    model = build_model()
    #print(model.summary())

    if(city == 'sj'):
      EPOCHS = 680
    elif(city == 'iq'):
      EPOCHS = 180

    history = model.fit(
      normed_train_data, train_labels,
      epochs=EPOCHS, validation_split = 0.2, verbose=0,
      shuffle=False, callbacks=[tfdocs.modeling.EpochDots()])


    test_predictions = model.predict(normed_test_data).flatten()
    test_predictions = np.nan_to_num(test_predictions)
    #print(test_predictions)
    result = []
    for i in test_predictions:
        result.append(round(i))

    return result

def main():
    sj = res('sj')
    iq = res('iq')
    result = sj + iq
    print()
    #print(len(result))
    #print(result)
    df = pd.read_csv("../data/submission_format.csv", index_col=[0, 1, 2])
    df['total_cases'] = result
    df.to_csv('sub.csv')

if __name__ == '__main__':
    main()