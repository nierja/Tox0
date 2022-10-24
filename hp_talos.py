#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

import sklearn.preprocessing
from scipy import sparse
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import os
import tensorflow as tf
from tensorflow import keras
import talos
# assert(tf.test.is_gpu_available())
gpus = tf.config.list_physical_devices('GPU'); logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs available")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--cv", default=10, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--fp", default="maccs", type=str, help="Fingerprint to use")
parser.add_argument("--pca", default=0, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--param_fraction", default=1.0, type=lambda x:int(x) if x.isdigit() else float(x), help="Fraction of all paremeter configurations to try out")


def main(args: argparse.Namespace) -> list:
    # We are training a model.
    np.random.seed(args.seed)

    print(args.fp, args.target)
    # load the training data, perform data cleaning and convert it into a numpy array
    df = pd.read_csv(f"Tox21_data/{args.target}/{args.target}_{args.fp}.data")
    df.replace([np.inf, -np.inf], np.nan, inplace=True, ); df.dropna(inplace=True)
    data, target = df.iloc[:, 0:-2].to_numpy(), df.iloc[:, -1].to_numpy()

    # perfoms the PCA transformation to R^2 space
    if args.pca:
        transformer = IncrementalPCA(n_components=args.pca)
        data = sparse.csr_matrix(data)
        data = transformer.fit_transform(data)

    # splitting dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # train a model on the given dataset and store it in 'model'.
    scaler = sklearn.preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # parameter dict
    p = {
        'activation': ['relu'],
        'first_neuron': [50, 100, data.shape[1]//4, data.shape[1]//2],
        'first_dropout': [0.0, 0.5],
        'second_neuron': [50, 100, data.shape[1]//4, data.shape[1]//2],
        'second_dropout': [0.0, 0.5],
        'third_neuron': [50, 100, data.shape[1]//4, data.shape[1]//2],
        'third_dropout': [0.0, 0.5],
        'batch_size': [64, 128],
        'epochs': [10,50,100], 
    }

    def tox_model(train_data, train_target, test_data, test_target, params):
        # define the model architecture
        model = keras.Sequential()
        model.add(keras.layers.Dense(data.shape[1], input_shape=(data.shape[1],), activation=params['activation']))
        model.add(keras.layers.Dense(params['first_neuron'], activation='relu'))
        model.add(keras.layers.Dropout(params['first_dropout']))
        model.add(keras.layers.Dense(params['second_neuron'], activation='relu'))
        model.add(keras.layers.Dropout(params['second_dropout']))
        model.add(keras.layers.Dense(params['third_neuron'], activation='relu'))
        model.add(keras.layers.Dropout(params['third_dropout']))
        model.add(keras.layers.Dense(args.n_classes-1, activation='sigmoid'))

        model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['AUC'])

        out = model.fit(x=train_data, 
                y=train_target,
                validation_data=[test_data, test_target],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0,
                callbacks=[talos.utils.early_stopper(params['epochs'])],
            )

        return out, model

    # perform the grid search, with early stopping
    scan_object = talos.Scan(
        x=train_data, 
        y=train_target, 
        params=p, 
        model=tox_model, 
        experiment_name=f'dnn_hparams_logs', 
        fraction_limit=args.param_fraction,
        print_params=False,
    )
    # perform analysis of the results
    analyze_object = talos.Analyze(scan_object)
        
    # # accessing the results data frame
    # print(scan_object.data)
    # # accessing epoch entropy values for each round
    # print(scan_object.learning_entropy)
    # # access the summary details
    # print(scan_object.details)
    # print(analyze_object.data)

    # get the best model and its parameters
    best_model = analyze_object.data[analyze_object.data.val_auc == analyze_object.data.val_auc.max()]
    best_params = p.copy()
    for key in best_params:
        best_params[key] = best_model.iloc[0][key]
    
    # perform k-fold crossvalidation
    # as in https://stackoverflow.com/questions/66695848/kfold-cross-validation-in-tensorflow
    auc_scores = []
    for kfold, (train, test) in enumerate(KFold(n_splits=args.cv, 
                                    shuffle=True).split(data, target)):
        # clear the session 
        tf.keras.backend.clear_session()

        # get the model
        _, seq_model = tox_model(train_data, train_target, test_data, test_target, best_params)

        # run the model 
        seq_model.fit(
            data[train], 
            target[train],
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            validation_data=(data[test], target[test]),
            verbose=0,
        )
        print(
            np.mean(seq_model.history.history['val_auc']), 
            np.std(seq_model.history.history['val_auc']), 
            seq_model.history.history['val_auc'],
        )
        auc_scores.extend(seq_model.history.history['val_auc'])
        # seq_model.save_weights(f'wg_{args.cv}.txt')

    # log data into a csv file
    file_path = f'./Results/talos_hp_results_{args.target}.csv'
    if not os.path.isfile(file_path): 
        # create a csv header if the file doesn't exist
        with open(file_path, 'w') as f:
            print("fp;pca;best_val_auc;crossval_auc;crossval_auc_std;best_params", file=f)

    with open(file_path, 'a') as f:
        print(f"{args.fp};{args.pca};{analyze_object.data.val_auc.max()};{np.array(auc_scores).mean()};{np.array(auc_scores).std()};{best_params}", file=f)
    
    # print to stdout as well
    print(f"fp={args.fp}, pca={args.pca}, val_auc={analyze_object.data.val_auc.max()}, best_params={best_params}")
    print(f"{args.cv}-fold crossvalidation auc = {np.array(auc_scores).mean()} +- {np.array(auc_scores).std()}")
    return 0

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)