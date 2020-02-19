import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
import math
import string
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


CLASS_COLUMN = "classname"
FILE_COLUMN = "filename"

# Utility function to build a class model in the form of a list of dictionaries.
# Each dictionary represents a row of the DataFrame {classname, filename, word1, word2, ...}
# Only the words that appear in the current document are added to the dictionary,
# the pandas DataFrame will automatically add the columns and fill the missing values with NaNs.
def __build_class_model(files, class_name, corpus_list):
	for filename in files:
		document_dict = {CLASS_COLUMN : class_name, FILE_COLUMN : filename}
		filestring = ""
		with open(filename, "r") as thefile:
			for line in thefile:
				filestring += "\n" + line
		text = filestring.lower().split() # lowercase so the words are not treated differently if they appear at the beginning of a sentence.
		text = [word for word in text if word not in string.punctuation] # removes punctuation marks
		document_dict.update(Counter(text))
		corpus_list.append(document_dict)
	return corpus_list

# folder1 and folder2 contain the path to the folders we want to use as corpus.
# The name of the folder is used as name for the relative class.
# n is an optional parameter we can use to change the minimum number of occurrences to consider.
def part1_load(folder1, folder2, n=100):
	allfiles1 = glob("{}/*.txt".format(folder1))
	allfiles2 = glob("{}/*.txt".format(folder2))
	corpus_list = __build_class_model(allfiles1, folder1, [])
	corpus_list = __build_class_model(allfiles2, folder2, corpus_list)
	corpus_df = pd.DataFrame(corpus_list)
	corpus_df = corpus_df.fillna(0)
	corpus_df = corpus_df.set_index([CLASS_COLUMN, FILE_COLUMN]) # Changes the shape of the DataFrame to consider only the columns relative to words
	corpus_df = corpus_df.loc[:, corpus_df.sum(axis=0) > n]
	return corpus_df.reset_index()

def part2_vis(df, m=10):
	assert isinstance(df, pd.DataFrame)

	table = pd.pivot_table(df, index=[CLASS_COLUMN], aggfunc=np.sum) # Pivoting the table on the class_column, we obtain the total frequencies of a word for that class in the corpus
	m_columns = table.sum(axis=0).sort_values(ascending=False)[:m]
	plot_table = table.loc[:, m_columns.index].T # Transpose the DataFrame to obtain the desired visualization

	return plot_table.plot(kind="bar")

def part3_tfidf(df):
	assert isinstance(df, pd.DataFrame)

	total_docs = len(df.index)
	words_only = df.select_dtypes(include='number') # Copies in a new DataFrame only the numerical columns (frequencies of the words)
	words_only = words_only.apply(lambda x: x * math.log(total_docs / (x > 0).sum()), axis=0) # Applies the tfidf to every value in the DataFrame
	df.update(words_only) # Applies the changes to the original DataFrame
	return df

# df is the DataFrame used to train and test the classifier
# The function uses a SVM classifier to create a model and use it on the test data to calculate accuracy
def part_bonus(df):
	svclassifier = SVC(kernel='linear')
	df = df.set_index(FILE_COLUMN)
	data = df.drop(CLASS_COLUMN, axis=1)
	labels = df[CLASS_COLUMN]
	training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size = 0.20) # Shuffles and splits the data in test and training set
	svclassifier.fit(training_data, training_labels)
	predicted_test_labels = svclassifier.predict(test_data)
	return accuracy_score(test_labels, predicted_test_labels)