# # Preparing Data
#
# In this project, we will use Yelp online reviews as a dataset, and mainly focus on building a filter by analyzing online text reviews. For more information of dataset, [find this].
#
# In this first notebook, we start process dataset by normalize and pretrain the dataset. Further notebooks will use different learning algorithms to train our refined dataset.
#
# Let's import some packages and data files.

import numpy as np
import torch
import torchtext
from torchtext import data
import spacy
import nltk

def regroup_split_data(data_folder):
    # ## Data Regroup & Split
    # From raw data files, for each sample they contains:
    # * metadata
    # ```
    # <user_id> <restaurant_id> <rating> <label> <date>
    # ```
    # * reviewContent
    # ```
    # <user_id> <restaurant_id> <date> <review>
    # ```
    #
    #
    # For this project we have to rearrange those sample as one sample dataset:
    content = np.loadtxt(data_folder+"reviewContent", dtype='str', delimiter="\t")
    data = np.loadtxt(data_folder+"metadata",
        dtype={'names': ('user_id', 'prod_id', 'rating', 'label', 'date'),
        'formats': (np.int_, np.int_, np.float, np.int_, '|S11')}, delimiter="\t")
    sc = content[:, 3].reshape(content.shape[0], 1)
    dt = np.array([data['user_id'], data['prod_id'], data['label'], data['rating']])
    dt = dt.T
    rst = np.hstack((dt, content))
    a = False
    b = False
    for i in range(rst.shape[0]):
        if float(rst[i][2]) < 0:
            if a:
                false_data = np.concatenate((false_data, np.array([rst[i]])))

            else:
                false_data = np.array([rst[i]])
                a = True
        else:
            if b:
                true_data = np.concatenate((true_data, np.array([rst[i]])))
            else:
                true_data = np.array([rst[i]])
                b = True

    np.random.shuffle(false_data)
    np.random.shuffle(true_data)
    train_size = round(false_data.shape[0] * 0.9)
    train = np.concatenate((false_data[:train_size],true_data[:train_size]))
    test = np.concatenate((false_data[train_size:],true_data[train_size:(false_data.shape[0])]))
    np.savetxt(data_folder+"input/train", train, fmt='%s', delimiter='\t')
    np.savetxt(data_folder+"input/test", test, fmt='%s', delimiter='\t')


if __name__ == "__main__":
    regroup_split_data("../data/")
