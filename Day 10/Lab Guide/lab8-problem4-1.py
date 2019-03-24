import re
from nltk.stem import PorterStemmer


def processEmail(email_content, vocabulary):
    """
    Preprocesses the body of an email and returns a list of indices of the words contained in the email.

    :param email_content:
    :param vocabulary:
    :return:
    """

    # Lower case
    email_content = email_content.lower()

    # Handle numbers
    email_content = re.sub("[0-9]+", "number", email_content)

    # Handle URLs
    email_content = re.sub("[http|https]://[^\s]*", "httpaddr", email_content)

    # Handle email address
    email_content = re.sub("[^\s]+@[^\s]+", "emailaddr", email_content)

    # Handle $ sign
    email_content = re.sub("[$]+", "dollar", email_content)

    # Strip all special characters
    special_characters = ["<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":"]
    for token in special_characters:
        email_content = email_content.replace(str(token), "")
    email_content = email_content.replace("\n", " ")

    # Stem the word
    stemmer = PorterStemmer()
    email_content = [stemmer.stem(token) for token in email_content.split(" ")]
    email_content = " ".join(email_content)

    # Process the email and return word_indices
    word_indices = []

    for token in email_content.split():
        for key, word in vocabulary.items():
            if token == word:
                word_indices.append(key)

    return word_indices


'''
Step 1: Read an email sample and the vocabulary. 
After that, store them in file_contents and vocabList.
'''

file_contents = open("data/lab8email_sample.txt", "r").read()
vocab_file = open("data/lab8vocab.txt", "r").read()

print('Content in the email sample:')
print(file_contents)

vocab_file = vocab_file.split('\n')

vocabulary = {}
for index, value in enumerate(vocab_file):
    if value is not '':
        ind_key, ind_value = value.split('\t')
        vocabulary[ind_key] = ind_value

print('Vocabulary list:')
print(vocabulary)

print('\n')


'''
Step 2: Find word indices which are corresponding to the email sample
'''

word_indices = processEmail(file_contents, vocabulary)

print('Word indices of the sample email is:')
print(word_indices)

