#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from tabulate import tabulate
import grid_search_parameters
import plotting

import sklearn.cluster
import sklearn.compose
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.decomposition import IncrementalPCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer 
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.random_projection import SparseRandomProjection


import os
# trick to import CUDA, following lines might need to be deleted/modified depending on your CUDA instalation
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
# os.add_dll_directory("C:/tools/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive/bin")
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
# assert(tf.test.is_gpu_available())
gpus = tf.config.list_physical_devices('GPU'); logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs available")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--cv", default=0, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--roc", default=True, type=bool, help="Plot the ROC_AUCs")
parser.add_argument("--vis", default=None, type=str, help="Visualisation type [Isomap|NCA|SRP|tSVD|TSNE]")
parser.add_argument("--pca", default=False, type=bool, help="Plot the PCAs")
parser.add_argument("--pca_comps", default=20, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--grid_search", default=False, type=bool, help="Perform hyperparameter optimisation, not supported for tf_cnn model")
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--model", default="tf_dnn", type=str, help="Model to use")
parser.add_argument("--scaler", default="MaxAbsScaler", type=str, help="defines scaler to preprocess data")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args: argparse.Namespace) -> list:
    # We are training a model.
    np.random.seed(args.seed)
    positive_PCA_features, positive_vis_features = [], []
    negative_PCA_features, negative_vis_features = [], []
    results, y_true, y_predicted_proba, fprs, tprs = [], [], [], [], []

    fpdict_keys = ['maccs', 'ecfp0','ecfp2', 'ecfp4', 'fcfp2', 'fcfp4']
    # fpdict_keys = ['ecfp4']
    # fpdict_keys = ['dist_2D' ,'dist_3D', 'adjac', 'inv_dist_2D', 'inv_dist_3D', 'Laplacian']
    # fpdict_keys = [ 'rdkit_descr', 'ecfp0','ecfp2', 'ecfp4', 'fcfp2', 'fcfp4', 'CMat_full', 'CMat_400', 'dist_2D' ,
    #               'dist_3D', 'adjac', 'Laplacian', ]
    # fpdict_keys = ['ecfp0','ecfp2', 'ecfp4', 'ecfp6', 'fcfp2', 'fcfp4', 'fcfp6',
    #               'maccs', 'hashap', 'hashtt', 'avalon', 'rdk5', 'rdk6', 'rdk7',
    #               'CMat_full', 'CMat_400', 'CMat_600', 'eigenvals', 'dist_2D' ,
    #               'dist_3D', 'balaban_2D', 'balaban_3D', 'adjac', 'Laplacian', ]

    for fp_name in fpdict_keys:
        print(fp_name, args.model, args.target)
        # load the training data, perform data cleaning and convert it into a numpy array
        df = pd.read_csv(f"Tox21_data/{args.target}/{args.target}_{fp_name}.data")
        df.replace([np.inf, -np.inf], np.nan, inplace=True, ); df.dropna(inplace=True)
        data, target = df.iloc[:, 0:-2].to_numpy(), df.iloc[:, -1].to_numpy()
        # imputer = SimpleImputer(missing_values=np.nan,strategy = "mean")
        # data = imputer.fit_transform(data)

        # perfoms the PCA transformation to R^2 space
        if args.pca:
            transformer = IncrementalPCA(n_components=args.pca_comps)
            data = sparse.csr_matrix(data)
            data = transformer.fit_transform(data)
            if args.pca_comps == 2:
                positive_PCA_features.append(data[target[:] == 1])
                negative_PCA_features.append(data[target[:] == 0])

        # splitting dataset into a train set and a test set.
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=args.test_size, random_state=args.seed)

        # train a model on the given dataset and store it in 'model'.
        if args.model in ["most_frequent", "stratified"]:
            model = sklearn.dummy.DummyClassifier(strategy=args.model)
        elif args.model == "gbt":
            model = sklearn.ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=20, verbose=1)
        elif args.model == "tf_dnn":
            scaler = sklearn.preprocessing.StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(data.shape[1], input_shape=(data.shape[1],), activation='relu'),
                tf.keras.layers.Dense(data.shape[1]//2, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(data.shape[1]//2, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],)
        elif args.model == "tf_rnn":
            VOCAB_SIZE = 100
            encoder = tf.keras.layers.TextVectorization(
                max_tokens=VOCAB_SIZE,
                split="character",
            )
            encoder.adapt(train_data.map(lambda text, label: text))

            model = tf.keras.Sequential([
                encoder,
                tf.keras.layers.Embedding(
                    input_dim=len(encoder.get_vocabulary()),
                    output_dim=64,
                    # Use masking to handle the variable sequence lengths
                    mask_zero=True),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])


            model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=['accuracy'])
            
            """## Train the model"""
            
            history = model.fit(train_data, epochs=10,
                                validation_data=test_data,
                                validation_steps=30)

            test_loss, test_acc = model.evaluate(test_data)
            print(test_loss, test_acc)
        else:
            if args.model == "lr":
                model = [
                    ("lr_cv", sklearn.linear_model.LogisticRegressionCV(max_iter=100)),
                ]
            elif args.model == "svm":
                model = [
                    ("svm", sklearn.svm.SVC(max_iter=100, probability=True, verbose=1, kernel="linear")),
                ]
            elif args.model == "adalr":
                model = [
                    ("ada_lr_cv", sklearn.ensemble.AdaBoostClassifier(sklearn.linear_model.LogisticRegression(C=1), n_estimators=5)),
                ]
            elif args.model == "baglr":
                model = [
                    ("bag_lr_cv", sklearn.ensemble.BaggingClassifier(sklearn.linear_model.LogisticRegression(C=1), n_estimators=5)),
                ]
            elif args.model == "badlr":
              model = [("lr", sklearn.linear_model.LogisticRegression())]
            elif args.model == "mlp":
                model = [
                    ("mlp", sklearn.neural_network.MLPClassifier(tol=0, learning_rate_init=0.01, max_iter=20, hidden_layer_sizes=(100), activation="relu", solver="adam", verbose=1)),
                ]

            # int_columns = []
            float_columns = list(range(0,train_data.shape[1]))
            # print(float_columns)
            if args.scaler == "StandardScaler": 
                model = sklearn.pipeline.Pipeline([
                    ("preprocess", sklearn.compose.ColumnTransformer([
                        ("scaler", sklearn.preprocessing.StandardScaler(), float_columns),
                    ]))
                ] + model)
            if args.scaler == "MinMaxScaler": 
                model = sklearn.pipeline.Pipeline([
                    ("preprocess", sklearn.compose.ColumnTransformer([
                        ("scaler", sklearn.preprocessing.MinMaxScaler(), float_columns),
                    ]))
                ] + model)
            if args.scaler == "MaxAbsScaler": 
                model = sklearn.pipeline.Pipeline([
                    ("preprocess", sklearn.compose.ColumnTransformer([
                        ("scaler", sklearn.preprocessing.MaxAbsScaler(), float_columns),
                    ]))
                ] + model)

        if args.cv:
            scores = sklearn.model_selection.cross_val_score(model, train_data, train_target, cv=args.cv)
            print("Cross-validation with {} folds: {:.2f} +-{:.2f}".format(args.cv, 100 * scores.mean(), 100 * scores.std()))

        # We now fit the constructed model
        # we can either search for the hyperparameters, or fit the model with default hyperparameters
        if args.grid_search:
            parameters = grid_search_parameters.get_grid_search_parameters(args.model)
            grid_search = GridSearchCV(model, parameters, n_jobs=-1)
            grid_search.fit(train_data, train_target)
            print(f"\nBest hyperparameters:\n{grid_search.best_params_}\n")
            test_predictions = grid_search.best_estimator_.predict(test_data)
            test_proba = grid_search.best_estimator_.predict_proba(test_data)
        elif args.model == "tf_dnn":
            # tensorflow SNN model needs to be trained differently than sklearn models
            model.fit(train_data, train_target, epochs=5, batch_size=4)
            predictions = model.predict(test_data)
            test_predictions = (predictions > 0.5).astype("int32")
            _ = np.ones((predictions.shape))
            test_proba = np.column_stack((_, predictions))
            test_proba[:, 0] -= test_proba[:, 1]
            # print(test_proba, test_predictions)
        else:
            # fitting sklearn models
            model.fit(train_data, train_target)
            test_predictions = model.predict(test_data)
            test_proba = model.predict_proba(test_data)

        accuracy = accuracy_score(test_target, test_predictions)
        balanced_accuracy = balanced_accuracy_score(test_target, test_predictions)

        ###############################################################
        # w = model["svm"].coef_
        # b = model["svm"].intercept_
        # ind = np.argpartition(w, -10)[0][-10:]
        # # print(w.shape, b.shape)
        # # print(w, b)
        # print(list(ind))
        # print(list(w[0,ind]))
        ###############################################################

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(args.n_classes):
            fpr[i], tpr[i], _ = roc_curve(test_target.ravel(), test_proba[:, i].ravel())
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area - micro-average NOT OK
        fpr["micro"], tpr["micro"], _ = roc_curve(test_target.ravel(), (test_proba[:,1]).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # store data for future plotting and console output
        results.append((fp_name, accuracy, balanced_accuracy, roc_auc["micro"])); 
        y_true.append(test_target); y_predicted_proba.append(test_proba)
        fprs.append(fpr["micro"]); tprs.append(tpr["micro"])

        # perfoms selected projection of training data to R^2 space
        if args.vis != None:
            # load the training data anew, since they were distorted with PCA
            df = pd.read_csv(f"Tox21_data/{args.target}/{args.target}_{fp_name}.data")
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
            data, target = df.iloc[:, 0:-2].to_numpy(), df.iloc[:, -1].to_numpy()
            imputer = SimpleImputer(missing_values=np.nan,strategy = "mean") 
            data = imputer.fit_transform(data)

            if args.vis == "Isomap": data = Isomap(n_components=args.n_classes).fit_transform(data)
            if args.vis == "NCA": data = NeighborhoodComponentsAnalysis(n_components=args.n_classes, init="pca",).fit_transform(data, target)
            if args.vis == "SRP": data = SparseRandomProjection(n_components=args.n_classes,).fit_transform(data)
            if args.vis == "TSNE": data = TSNE(n_components=args.n_classes, learning_rate='auto', init='random').fit_transform(data)
            if args.vis == "tSVD": data = TruncatedSVD(n_components=args.n_classes).fit_transform(data)
            positive_vis_features.append(data[target[:] == 1])
            negative_vis_features.append(data[target[:] == 0])

    if args.roc: plotting.plot_ROCs(fprs, tprs, fpdict_keys, nrows=2, ncols=3, model=args.model, target=args.target)
    if args.vis != None: plotting.plot_DimReds(args.vis, positive_vis_features, negative_vis_features, fpdict_keys, nrows=2, ncols=3, model=args.model, target=args.target)
    if args.pca and args.pca_comps == 2: plotting.plot_DimReds("PCA", positive_PCA_features, negative_PCA_features, fpdict_keys, nrows=2, ncols=3, model=args.model, target=args.target)

    return results


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    results = main(args)
    with open(f"output_{args.target}.txt", "a") as output_file:
        table = tabulate(results, headers=["fp_name", "acc", "balanced_acc", "roc"], floatfmt=(None, '.4f', '.2f',))
        output_file.write(f"\n{args.model} - {args.target} \nTest size = {args.test_size}\n" + table)
    print(table)
