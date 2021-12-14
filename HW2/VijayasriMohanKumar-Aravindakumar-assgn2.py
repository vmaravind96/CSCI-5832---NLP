#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ## Part 1: Featurizing the reviews

# In[1]:


# All imports go here
import numpy as np
import random


# In[2]:


# Read the files for train data
# Split each review into review ID and review text and store in Numpy Array
def read_reviews_from_file(filePath):
    """
    Reads the reviews from given file and performs case conversion
    """
    with open(filePath) as fp:
        reviews = fp.readlines()
        reviews = np.array([review.strip().split('\t') for review in reviews])
        # Conversion to lower case
        reviews[:, 1] = np.char.lower(reviews[:, 1])
    return reviews


# In[3]:


# Read the List of Positive and negative words
def read_words_from_file(filePath):
    """
    Reads and returns set of words read from a file
    """
    with open(filePath) as fp:
        words = fp.readlines()
        words = set([word.strip() for word in words])
    return words


# In[4]:


# Initialize the pronouns 
all_pronouns = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]


# In[5]:


# Read the positive and negative reviews
positive_reviews = read_reviews_from_file("hotelPosT-train.txt")
negative_reviews = read_reviews_from_file("hotelNegT-train.txt")

# Print the statistics for train data set 
print("Positive reviews:", positive_reviews.shape)
print("Negative reviews:", negative_reviews.shape)


# In[6]:


# Read Positive and Negative words
positive_words = read_words_from_file("positive-words.txt")
negative_words = read_words_from_file("negative-words.txt")

# Print the statistics for Postive, Negative and pronouns
print("Positive words:", len(positive_words))
print("Negative words:", len(negative_words))
print("Pronouns:", len(all_pronouns))


# In[7]:


# Helper methods
def get_word_counts(stripped_words):
    """
    Returns the count of positive words in the review
    """
    # Count the positive and negative words and pronouns
    positives = [word for word in stripped_words if word in positive_words]
    negatives = [word for word in stripped_words if word in negative_words]
    cur_pronouns = [word for word in stripped_words if word in all_pronouns]
    return len(positives), len(negatives), len(cur_pronouns)

def check_if_no_present(stripped_words):
    """
    Returns 1 if the word no is present in the review else 0
    """
    is_present = any([word=="no" for word in stripped_words])
    return 1 if is_present else 0

def check_if_exclamation_present(current_words):
    """
    Returns 1 if ! is present in the review else 0
    """
    is_present = any([word.endswith('!') for word in current_words])
    return 1 if is_present else 0

def process_reviews(reviews, polarity=None):
    """
    Process reviews of a given polarity and return the processed results
    
    Params: 
        reviews: Reviews obtained from the training data of a particular polarity
        polarity: 1 for Positive and 0 for negative reviews
    """
    processed_reviews = []
    for idx, row in enumerate(reviews):
        current_words = row[1].split(" ")
    
        # Strip the special characters from the words like [,.!]
        stripped_words = [word.strip("!.,") for word in current_words]

        # Get Positive and negative word counts and pronoun counts
        pos_count, neg_count, pronoun_count = get_word_counts(stripped_words)

        # Check if "no" present in the review
        no_present = check_if_no_present(stripped_words)

        # Check if "!" present in the review
        # Note: We should not pass the stripped words for this calculation
        exclamation_present = check_if_exclamation_present(current_words)

        # Get Log of the word count of the document
        log_word_count = round(np.log(len(current_words)),2)

        # Current result
        # ID, PosCount, NegCount, NoPresent, PronounCount, ExclamationPresent, LogWordCounts, ReviewPolarity
        if polarity is not None:
            current_result = [row[0], pos_count, neg_count, no_present, pronoun_count, exclamation_present, log_word_count, polarity]
        else:
            current_result = [row[0], pos_count, neg_count, no_present, pronoun_count, exclamation_present, log_word_count]
        
        # Append the current result to processed o/p
        processed_reviews.append(current_result)
    
    return np.array(processed_reviews, dtype='object')
    


# In[8]:


# Process Positive and negative reviews
processed_positives = process_reviews(positive_reviews, 1)
processed_negatives = process_reviews(negative_reviews, 0)


# In[9]:


# Print the statistics for processed reviews
print("Processed Positives:", processed_positives.shape)
print("Processed Negatives:", processed_negatives.shape)


# In[10]:


# Concatenate Positive and Negative processed output
all_processed_reviews = np.concatenate((processed_positives, processed_negatives), axis=0)


# In[11]:


# Print the statistics of the combined reviews
print("Final Processed Reviews:", all_processed_reviews.shape)


# In[12]:


# Write the processed output to a CSV file
formatter = ('%s,%d,%d,%d,%d,%d,%.2f,%d')
np.savetxt("VijayasriMohanKumar-Aravindakumar-assgn2-part1.csv", all_processed_reviews, delimiter=",", fmt=formatter)


# ## Part 2 - Training, DevTesting and Testing

# In[13]:


# Helper methods:

def convert_data_type(data):
    """
    Change the data type of Columns in the CSV data
    """
    # Initialize necessary params for conversion
    new_data = np.empty(data.shape, dtype=object)
    integer_indices = [1, 2, 3, 4, 5, 7]
    float_indices = [6]
    string_indices = [0]
    
    for idx in range(data.shape[1]):
        if idx in string_indices:
            new_data[:, 0] = data[:, 0]
        elif idx in float_indices:
            new_data[:, 6] = data[:, 6].astype(np.dtype('float64'))
        elif idx in integer_indices:
            new_data[:, idx] = data[:, idx].astype(np.dtype('int32'))
    
    return new_data

def read_processed_train_data(filePath):
    # Read the processed train data from Part 1
    with open(filePath) as fp:
        csv_data = fp.readlines()
        csv_data = np.array([line.strip().split(',') for line in csv_data])
        csv_data = convert_data_type(csv_data)
    return csv_data


# In[14]:


# Read the processed training data
csv_data = read_processed_train_data("VijayasriMohanKumar-Aravindakumar-assgn2-part1.csv")

# Print the train data statistics
print("Train Data:", csv_data.shape)


# In[15]:


# Split Train data into 80:20 ratio for training and dev-testing
train_ratio = 0.80
total_size = len(csv_data)
train_size = int(total_size * train_ratio)
print("Desired Train data set size:",train_size)

# Step 1: Randomize the csv_data to get mixture of positive and negative reviews
train_indices = random.sample(range(total_size), train_size)
dev_test_indices = list(set(range(total_size)).difference(train_indices))
print("Total entries for Train :", len(train_indices))
print("Total entries for Dev-Test:", len(dev_test_indices))

# Check if there is any overlap between train and dev test data
print("Intersection between Train and Dev-test:", len(set(train_indices).intersection(dev_test_indices)))


# In[16]:


# Get the entries corresponding to the obtained indices
train_data = np.take(csv_data, indices=train_indices, axis=0)
dev_test_data = np.take(csv_data, indices=dev_test_indices, axis=0)


# In[17]:


# Print the statistics for train and dev_test
print("Train Data:", train_data.shape)
print("Dev Test Data:", dev_test_data.shape)

pos_count = list(train_data[:,7]).count(1)
print("Postive and negative samples in Train     =>", f"Pos: {pos_count}", f"Neg: {len(train_data)-pos_count}")
pos_count = list(dev_test_data[:,7]).count(1)
print("Postive and negative samples in Dev Test  =>", f"Pos: {pos_count}", f"Neg: {len(dev_test_data)-pos_count}")


# In[18]:


# Split into X_train, Y_train and X_dev_test, Y_dev_test
def train_test_split(train_data, dev_test_data):
    """
    Splits the data into X_train, Y_train and x_dev_test, y_dev_test
    """
    X_train, Y_train = train_data[:, :7], train_data[:, 7]
    X_dev_test, Y_dev_test = dev_test_data[:, :7], dev_test_data[:, 7]
    
    # Add a dummy feature to train for bias term
    X_train = np.append(X_train, np.array([1]*len(X_train)).reshape((-1,1)), axis=1)
    X_dev_test = np.append(X_dev_test, np.array([1]*len(X_dev_test)).reshape(-1,1), axis=1)
    
    return X_train, Y_train, X_dev_test, Y_dev_test


# In[19]:


# Get the train and dev-test split
X_train, Y_train, X_dTest, Y_dTest = train_test_split(train_data, dev_test_data)
print(X_train.shape, Y_train.shape, X_dTest.shape, Y_dTest.shape)


# ### SGD Implementation

# In[20]:


# Helpers
def get_class_score(z):
    """
    Returns Class score
    """
    # To eliminate numerical underflow increase the byte size.
    score = np.float128(1/(1+np.exp(-z)))
    return score

def perform_SGD(X_train, Y_train, learning_rate=0.01, bias=0.1, epochs=15000):
    """
    Runs SGD and returns final weights and epochs
    """
    # Initialize Epoch
    epoch = 0
    
    # Weights
    weights = [0]*6 + [bias]
    weights = np.array(weights)
    print("Initial weights:", weights, "Shape:", weights.shape)
    
    # Take a copy of the X_train and Y_train
    X_train_copy = X_train * 1
    Y_train_copy = Y_train * 1
    
    while epoch <= epochs:
    
        if len(X_train_copy)==0:
            X_train_copy = X_train * 1
            Y_train_copy = Y_train * 1

        # Choose a random point from train data
        random_idx = random.randint(0, len(X_train_copy)-1)

        # Get features and label corresponding to the index
        features = X_train_copy[random_idx]
        actual_score = Y_train_copy[random_idx]

        # Remove the entries in current index
        # To implement Random sampling without replacement
        # This ensures all the data points have been tried atleast once across all epoch.
        X_train_copy = np.delete(X_train_copy, random_idx, axis=0)
        Y_train_copy = np.delete(Y_train_copy, random_idx)

        # Compute the gradient
        # Features will also have ID which we exclude in gradient calculation
        z = np.dot(weights, features[1:])
        predicted_score = get_class_score(z)

        # Gradient = (predicted-actual)*features
        gradient = (predicted_score - actual_score) * features[1:]

        # Get updated weights
        new_weights = weights - learning_rate * gradient

        # Reassign the weights for next cycle
        weights = new_weights

        # Increment the epoch
        epoch += 1

    # end while
    
    return weights, epoch


# In[21]:


# Run SGD
weights, epoch = perform_SGD(X_train, Y_train, learning_rate=0.01, bias=0.1, epochs=20000)

# Print epochs and final weights
print("Epochs completed:", epoch)
print("Final weights:", weights)


# ### Validating the accuracy with Dev Test set

# In[22]:


# Helpers

# Cross entropy loss calculator
def cross_entropy_loss(y, y_hat):
    loss = (y * np.log(y_hat)) + ((1-y) * np.log(1-y_hat))
    return loss

def get_accuracy_and_loss(X_dTest, Y_dTest):
    """
    Compute the accuracy and loss
    """
    # Compute the cross entropy loss
    losses = []
    correct_preds = 0

    for idx, test_feature in enumerate(X_dTest):
    
        # Get the label
        y = Y_dTest[idx]

        # Compute the probability
        z = np.dot(weights, test_feature[1:])
        y_hat = get_class_score(z)

        pred = 1 if y_hat > 0.5 else 0
        if y == pred:
            correct_preds += 1

        # Compute the loss
        cur_loss = cross_entropy_loss(y, y_hat)

        # Append to all loss
        losses.append(cur_loss)
    
    N = len(X_dTest)
    average_loss = -(1/N) * sum(losses)
    accuracy = (correct_preds/N)*100
    
    return correct_preds, average_loss, accuracy


# In[23]:


# Test the model and DEV dataset and report the accuracy
print("Metrics on Dev Test Data Set:")
correct_preds, cp_loss, accuracy = get_accuracy_and_loss(X_dTest, Y_dTest)
print("Total correct classifications:", correct_preds)
print("Accuracy:", accuracy)
print("Cross Entropy loss:", cp_loss)


# In[24]:


# Metrics on Train Data Set
print("Metrics on Train Data Set:")
correct_preds, cp_loss, accuracy = get_accuracy_and_loss(X_train, Y_train)
print("Total correct classifications:", correct_preds)
print("Accuracy:", accuracy)
print("Cross Entropy loss:", cp_loss)


# ### Evaluating the model on Actual Test Data

# In[25]:


# Read the test reviews from the file
# Replace the file name on getting the test data
test_reviews = read_reviews_from_file("HW2-testset.txt")
print("Test Reviews:", test_reviews.shape)


# In[26]:


# Featurize the reviews 
X_test = process_reviews(test_reviews)
print("Featurized Test Data:", X_test.shape)


# In[27]:


# For each test data compute the probability a using the weights obtained above in SGD
def predict(X_test, weights):
    """
    Predicts the polarity based on the weights obtained
    """
    # Add 1 as Column for bias
    X_test = np.append(X_test, np.array([1]*len(X_test)).reshape((-1,1)), axis=1)
    print("Shape after adding Bias column:", X_test.shape)
    
    result = []
    for review in X_test:
        # Get Predicted Score
        predicted_score = get_class_score(np.dot(weights, review[1:]))
        
        # Decide class based on Probability
        if (predicted_score <= 0.5):
            class_label = "NEG"
        else:
            class_label = "POS"
        
        # Append to result
        result.append([review[0], class_label])
        
    result = np.array(result, dtype='object')
    return result


# In[28]:


# Predict the polarity of the reviews
predicted_polarities = predict(X_test, weights)
print("Predicted Polarities:", predicted_polarities.shape)


# In[29]:


# Write to a file in the given format
np.savetxt("VijayasriMohanKumar-Aravindakumar-assgn2-out.txt", predicted_polarities, delimiter="\t", fmt="%s")

