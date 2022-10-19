#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

import sklearn.preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import talos
# assert(tf.test.is_gpu_available())
gpus = tf.config.list_physical_devices('GPU'); logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs available")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--cv", default=0, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--pca", default=False, type=bool, help="Plot the PCAs")
parser.add_argument("--pca_comps", default=20, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--grid_search", default=False, type=bool, help="Perform hyperparameter optimisation, not supported for tf_cnn model")
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args: argparse.Namespace) -> list:
    # We are training a model.
    np.random.seed(args.seed)
    results, y_true, y_predicted_proba, fprs, tprs = [], [], [], [], []

    # fpdict_keys = ['rdkit_descr', 'ecfp0','ecfp2', 'ecfp4', 'fcfp2', 'fcfp4']
    fpdict_keys = ['ecfp4']

    for fp_name in fpdict_keys:
        print(fp_name, args.target)
        # load the training data, perform data cleaning and convert it into a numpy array
        df = pd.read_csv(f"Tox21_data/{args.target}/{args.target}_{fp_name}.data")
        df.replace([np.inf, -np.inf], np.nan, inplace=True, ); df.dropna(inplace=True)
        data, target = df.iloc[:, 0:-2].to_numpy(), df.iloc[:, -1].to_numpy()

        # splitting dataset into a train set and a test set.
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=args.test_size, random_state=args.seed)

        # train a model on the given dataset and store it in 'model'.
        scaler = sklearn.preprocessing.StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        p = {
            'first_neuron': [50, 100, 150],
            'activation': ['relu', 'elu'],
            'batch_size': [32, 64],
        }

        def tox_model(train_data, train_target, test_data, test_target, params):
            model = keras.Sequential()
            model.add(keras.layers.Dense(data.shape[1], input_shape=(data.shape[1],), activation=params['activation']))

            model.add(keras.layers.Dense(params['first_neuron'], activation='relu'))
            model.add(keras.layers.Dense(1, activation='sigmoid'))

            model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['AUC'])

            out = model.fit(x=train_data, 
                    y=train_target,
                    validation_data=[test_data, test_target],
                    epochs=10,
                    batch_size=params['batch_size'],
                    verbose=0)

            return out, model

        scan_object = talos.Scan(x=train_data, y=train_target, params=p, model=tox_model, experiment_name='Tox21_dnn_hparams')
        
        # accessing the results data frame
        print(scan_object.data.head())

        # accessing epoch entropy values for each round
        print(scan_object.learning_entropy)

        # access the summary details
        print(scan_object.details)

        analyze_object = talos.Analyze(scan_object)
        print(analyze_object.data)

    return


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    