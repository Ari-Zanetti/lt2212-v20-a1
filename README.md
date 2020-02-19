# LT2212 V20 Assignment 1

Part 1 - Convert the data into a dataframe

Every document is converted into a list of strings using the function split(). 
The words are lowercased so that there is no difference if they appear at the beginning or in the middle of a sentence.  
This could, eventually, be a problem in presence of acronyms or proper names. 
Before counting the occurrences of each word, the strings that appear in the string.punctuation list are removed.


Part 2 - visualize

To obtain the desired visualization, the DataFrame is pivoted on the class column, specifying "sum" as the aggregation function. 
For every word (the column containing the name of the documents is not considered because it is not numeric), we have two values:
	- the count of occurrences of the words in the crude class; 
	- the count of occurrences of the words in the grain class.


Part 3 - tf-idf

To calculate tf-idf we need 3 different values:
	- the total number of documents, that we can obtain looking at the number of rows in the DataFrame; 
	- the frequency of a word in a document, that we already have being the value of every "word column" (not class name or file name) in the DataFrame; 
	- the number of documents containing the word, that we can obtain by counting the number of rows with a value greater than 0.


Part 4 - visualize again

We notice that, compared to the chart in part2, there are less function words, like articles ("the", "a"), or prepositions ("of", "in"), and more content words, specific of different types of documents, like "oil" for the class "crude", that was the tenth word with the raw count, and "wheat" for the class "grain".


Part Bonus

The function part_bonus(df) takes a DataFrame as input, separates the class column from the others creating two DataFrames: "data" and "labels". 
Then, it uses the function "train_test_split" contained in the sklearn module to shuffle the DataFrame and split it into two parts to use as train test and test set, and it creates a SVM Classifier using the training data as model. 
It returns a value corresponding to the accuracy calculated over the test set with the function "accuracy_score" of the sklearn module. 
Running the classification several times, the accuracy ranges between 94 and 97 % for the raw counts and between 95 and 98 % for the tfidf counts. 
This is because, with tfidf, the words that are more specific for a document have a higher weight, so the classifier can create a better model of the training set.

