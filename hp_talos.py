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

gpus = tf.config.list_physical_devices('GPU'); logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs available")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--n_layers", default=3, type=int, help="Number of hidden layers")
parser.add_argument("--cv", default=3, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--NN_type", default="DNN", type=str, help="Type of a NN architecture")
parser.add_argument("--fp", default="ecfp4", type=str, help="Fingerprint to use")
parser.add_argument("--pca", default=20, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--param_fraction", default=0.001, type=lambda x:int(x) if x.isdigit() else float(x), help="Fraction of all paremeter configurations to try out")


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
    # for CNN, transform the data into 3D tensor
    if args.NN_type == "CNN":
        data.reshape(data.shape[0],data.shape[1],1)
        print(data.shape)

    # scale the data to have zero mean and unit variance
    scaler = sklearn.preprocessing.StandardScaler()
    data = scaler.fit_transform(data)

    # splitting dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # create parameter dict
    p = {
        'activation': ['relu'],
        'batch_size': [64, 128],
        'epochs': [10,50,100], 
    }
    
    # fill the dictionary up
    if args.NN_type == "DNN":
        for i in range(1, args.n_layers+1):
            p[f'hidden_layer_{i}'] = [50, 100, data.shape[1]//4, data.shape[1]//2]
            p[f'dropout_layer_{i}'] = [0.0, 0.5]
    if args.NN_type == "CNN":
        for i in range(1, args.n_layers+1):
            p[f'conv_layer_{i}_filter'] = [2*i, 4*i, 8*i]
            p[f'conv_layer_{i}_kernel'] = [3, 5]
        p['conv_hidden_layer'] = [data.shape[1]//8, data.shape[1]//4, data.shape[1]//2]
        p['conv_dropout'] = [0, 0.5]

    def tox_model(train_data, train_target, test_data, test_target, params):
        model = keras.Sequential()

        # define the model architecture
        if args.NN_type == "DNN":
            model.add(keras.layers.Dense(data.shape[1], input_shape=(data.shape[1],), activation=params['activation']))
            for i in range(1, args.n_layers+1):
                model.add(keras.layers.Dense(params[f'hidden_layer_{i}'], activation='relu'))
                model.add(keras.layers.Dropout(params[f'dropout_layer_{i}']))

        if args.NN_type == "CNN":
            model.add(keras.layers.Conv1D(
                params[f'conv_layer_1_filter'], 
                params[f'conv_layer_1_kernel'], 
                input_shape=(data.shape[1],1), 
                activation='relu',
                )
            )
            model.add(keras.layers.MaxPooling1D())
            for i in range(2, args.n_layers+1):
                model.add(keras.layers.Conv1D(
                    params[f'conv_layer_{i}_filter'], 
                    params[f'conv_layer_{i}_kernel'], 
                    activation='relu'),
                )
                model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(params['conv_hidden_layer'], activation=tf.nn.relu))
            model.add(keras.layers.Dropout(params['conv_dropout']))

        # add one neuron to the output layer to perform the binary classification
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

    # get the best model and its parameters
    best_model = analyze_object.data[analyze_object.data.val_auc == analyze_object.data.val_auc.max()]
    best_params = p.copy()
    for key in best_params:
        if "kernel" in key:
            best_params[key] = (int(best_model.iloc[0][key]))
        else:
            best_params[key] = best_model.iloc[0][key]
    print(best_params)

    # import visualkeras
    # _, model = tox_model(train_data, train_target, test_data, test_target, best_params)
    # visualkeras.layered_view(model).show() # display using your system viewer
    # visualkeras.layered_view(model, to_file='./Plots/visualkeras_output_CNN.png') # write to disk
    # visualkeras.layered_view(model, to_file='./Plots/visualkeras_output_CNN.png').show() # write and show
    
    # perform k-fold crossvalidation
    # as in https://stackoverflow.com/questions/66695848/kfold-cross-validation-in-tensorflow
    auc_scores = []
    for kfold, (train, test) in enumerate(KFold(n_splits=args.cv, 
                                    shuffle=True).split(data, target)):
        # clear the session 
        tf.keras.backend.clear_session()

        # get the model
        _, seq_model = tox_model(train_data, train_target, test_data, test_target, best_params)

        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

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