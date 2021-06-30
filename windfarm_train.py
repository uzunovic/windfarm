from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RFC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print("in")

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int, default=10)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, engine="python", header=None, delimiter='|') for file in input_files ]
    train_data = pd.concat(raw_data)

    # Labels are in the last column
    train_y = train_data.iloc[:, -1]
    train_X = train_data.iloc[:, :-1]

    # Here we support a single hyperparameter, 'max_ldepth'. 
    max_leaf_nodes = args.max_depth

    # Classifier to train the model.
    clf = RFC(max_depth=10, random_state=0)
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
