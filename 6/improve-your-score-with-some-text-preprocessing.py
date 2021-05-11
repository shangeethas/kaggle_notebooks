#!/usr/bin/env python
# coding: utf-8

# # Improve your Score with some Text Preprocessing
# 
# ## Updated version : 
#  > ###  https://www.kaggle.com/theoviel/improve-your-score-with-text-preprocessing-v2
# 
# 
# 
# 
# This kernel is an improved version of @Dieter's work.
# > https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
# 
# 
# It is the preprocessing I use for my current LB score, and it has helped improving it by a bit. Feel free to use it as well, but please upvote if you do. 
# 
# This is also how I caught a glimpse of spelling mistakes in the database.
# 
# #### Any feedback is appreciated ! 

# In[ ]:


import pandas as pd
import numpy as np
import operator 
import re


# ## Loading data

# In[ ]:


train = pd.read_csv("../input/train.csv").drop('target', axis=1)
test = pd.read_csv("../input/test.csv")
df = pd.concat([train ,test])

print("Number of texts: ", df.shape[0])


# ## Loading embeddings

# In[ ]:


def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index


# In[ ]:


glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'


# In[ ]:


print("Extracting GloVe embedding")
embed_glove = load_embed(glove)
print("Extracting Paragram embedding")
embed_paragram = load_embed(paragram)
print("Extracting FastText embedding")
embed_fasttext = load_embed(wiki_news)


# ## Vocabulary and Coverage functions
# > Again, check Dieter's work if you haven't, those are his.

# In[ ]:


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# In[ ]:


def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words


# ## Starting point

# In[ ]:


vocab = build_vocab(df['question_text'])


# In[ ]:


print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)


#  #### Paragram seems to have a significantly lower coverage. 
# >That's because it does not understand upper letters, let us lower our texts :

# In[ ]:


df['lowered_question'] = df['question_text'].apply(lambda x: x.lower())


# In[ ]:


vocab_low = build_vocab(df['lowered_question'])


# In[ ]:


print("Glove : ")
oov_glove = check_coverage(vocab_low, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab_low, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab_low, embed_fasttext)


# #### Better, but we lost a bit of information on the other embeddings.
# > Therer are words known that are known with upper letters and unknown without. Let us fix that :
# - word.lower() takes the embedding of word if word.lower() doesn't have an embedding

# In[ ]:


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")


# In[ ]:


print("Glove : ")
add_lower(embed_glove, vocab)
print("Paragram : ")
add_lower(embed_paragram, vocab)
print("FastText : ")
add_lower(embed_fasttext, vocab)


# In[ ]:


print("Glove : ")
oov_glove = check_coverage(vocab_low, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab_low, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab_low, embed_fasttext)


# ### What's wrong ?

# In[ ]:


oov_glove[:10]


# #### First faults appearing are : 
# - Contractions 
# - Words with punctuation in them
# 
# > Let us correct that.

# ## Contractions

# In[ ]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }


# In[ ]:


def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known


# In[ ]:


print("- Known Contractions -")
print("   Glove :")
print(known_contractions(embed_glove))
print("   Paragram :")
print(known_contractions(embed_paragram))
print("   FastText :")
print(known_contractions(embed_fasttext))


# #### FastText does not understand contractions
# > We use the map to replace them

# In[ ]:


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


# In[ ]:


df['treated_question'] = df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))


# In[ ]:


vocab = build_vocab(df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)


# ## Now, let us deal with special characters

# In[ ]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


# In[ ]:


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


# In[ ]:


print("Glove :")
print(unknown_punct(embed_glove, punct))
print("Paragram :")
print(unknown_punct(embed_paragram, punct))
print("FastText :")
print(unknown_punct(embed_fasttext, punct))


# #### FastText seems to have a better knowledge of special characters 
# > We use a map to replace unknown characters with known ones.
# 
# > We make sure there are spaces between words and punctuation
# 

# In[ ]:


punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }


# In[ ]:


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text


# In[ ]:


df['treated_question'] = df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))


# In[ ]:


vocab = build_vocab(df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)


# In[ ]:


oov_fasttext[:100]


# ### What's still missing ? 
# - Unknown words
# - Acronyms
# - Spelling mistakes

# ## We can correct manually most frequent mispells

# #### For example, here are some mistakes and their frequency
# - qoura : 85 times
# - mastrubation : 38 times
# - demonitisation : 30 times
# - …

# In[ ]:


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


# In[ ]:


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


# In[ ]:


df['treated_question'] = df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))


# In[ ]:


vocab = build_vocab(df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)


# ### That's all for now !
# 
# #### Improvement ideas: 
# > Replace acronyms with their meaning
# 
# > Replace unknown words with a more general term : 
#  - ex : fortnite, pubg -> video game
#  
#  ### *Thanks for reading ! *
