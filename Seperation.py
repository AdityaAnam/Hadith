#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#print(os.getcwd())
#print(os.listdir())

for root,dirs,files in os.walk("c:/Users/Aditya Pratap Singh/Desktop/Hadith/All Hadith Books"):
        print (root)
        print("FILES:")
        for filename in files:
            print (filename)
            print(os.path.join(root, filename))
            print ('-----------------------')
            
            
    #for filename in filenames:
        #print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd
import numpy as np
import re
import os

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer

get_ipython().system('pip install Tashaphyne')
import pyarabic.arabrepr
from tashaphyne.stemming import ArabicLightStemmer

import random
from sklearn.model_selection import train_test_split


# In[3]:


stop_words =['من',
 'في',
 'على',
 'و',
 'فى',
 'يا',
 'عن',
 'مع',
 'ان',
 'هو',
 'علي',
 'ما',
 'اللي',
 'كل',
 'بعد',
 'ده',
 'اليوم',
 'أن',
 'يوم',
 'انا',
 'إلى',
 'كان',
 'ايه',
 'اللى',
 'الى',
 'دي',
 'بين',
 'انت',
 'أنا',
 'حتى',
 'لما',
 'فيه',
 'هذا',
 'واحد',
 'احنا',
 'اي',
 'كده',
 'إن',
 'او',
 'أو',
 'عليه',
 'ف',
 'دى',
 'مين',
 'الي',
 'كانت',
 'أمام',
 'زي',
 'يكون',
 'خلال',
 'ع',
 'كنت',
 'هي',
 'فيها',
 'عند',
 'التي',
 'الذي',
 'قال',
 'هذه',
 'قد',
 'انه',
 'ريتويت',
 'بعض',
 'أول',
 'ايه',
 'الان',
 'أي',
 'منذ',
 'عليها',
 'له',
 'ال',
 'تم',
 'ب',
 'دة',
 'عليك',
 'اى',
 'كلها',
 'اللتى',
 'هى',
 'دا',
 'انك',
 'وهو',
 'ومن',
 'منك',
 'نحن',
 'زى',
 'أنت',
 'انهم',
 'معانا',
 'حتي',
 'وانا',
 'عنه',
 'إلي',
 'ونحن',
 'وانت',
 'منكم',
 'وان',
 'معاهم',
 'معايا',
 'وأنا',
 'عنها',
 'إنه',
 'اني',
 'معك',
 'اننا',
 'فيهم',
 'د',
 'انتا',
 'عنك',
 'وهى',
 'معا',
 'آن',
 'انتي',
 'وأنت',
 'وإن',
 'ومع',
 'وعن',
 'معاكم',
 'معاكو',
 'معاها',
 'وعليه',
 'وانتم',
 'وانتي',
 '¿',
 '|']


# In[4]:



def normalize(sentence):
 '''
 Argument:
     string of words
 return:
     string of words but standardize the words
 '''
 sentence = re.sub("[إأآا]", "ا", sentence)
 sentence = re.sub("ى", "ي", sentence)
 sentence = re.sub("ؤ", "ء", sentence)
 sentence = re.sub("ئ", "ء", sentence)
 sentence = re.sub("ة", "ه", sentence)
 sentence = re.sub("گ", "ك", sentence)
 return sentence


# In[5]:


def removing_ar_stopwords(text):
    """
        Here we remove all Arabic stop words
        
    """
      # if read it from file
#     ar_stopwords_list = open("arabic_stopwords.txt", "r") 
#     stop_words = ar_stopwords_list.read().split("\n")
#     stop_words = []
    original_words = []
    words = word_tokenize(text) # it works on one hadith not list
    for word in words:
        if word not in stop_words:
            original_words.append(word)
    filtered_sentence = " ".join(original_words)
    return filtered_sentence


# In[6]:


def clearReg(text):
    """
        This function for getting the normal values of out of lemmatization function
        that takse a string of dict as a 
        takes  : '{"result":["امر","ب","أخذ","ما","نهى","ه","انتهى"]}'
        return : ['امر أخذ ما نهى انتهى']
    """
    each_lemma_word = []
    each_lemma_sentence = []
    for hadith in text:
        matches = re.findall(r'\"(.+?)\"',hadith)
        for word in matches:
            if len(word) >= 2 and word !='result':
                each_lemma_word.append(word)
        each_lemma_sentence.append(" ".join(each_lemma_word))
        each_lemma_word.clear()
    return each_lemma_sentence


# In[7]:


def stemming_1(text):
    """
        This is first functoin for stemming and it's looks not good accurac, NLTK by ISRIStemmer.
    """
    st = ISRIStemmer()
    stemmend_words = []
    words = word_tokenize(text)
    for word in words:
        stemmend_words.append(st.stem(word))
    stemmed_sentence = " ".join(stemmend_words)
    return stemmed_sentence
        
    
    
def stemming_2(text):
    """
        This is Second functoin for stemming and it's looks good results, with built in library called Tashaphyne.
        The documentation here ==> https://pypi.org/project/Tashaphyne/
    
    """
    import pyarabic.arabrepr
    arepr = pyarabic.arabrepr.ArabicRepr()
    repr = arepr.repr

    from tashaphyne.stemming import ArabicLightStemmer
    ArListem = ArabicLightStemmer()

    hadiths_without_stop_words_and_with_normalization_and_with_stemming = []

    for hadith in hadiths_without_stop_words_and_with_normalization:
        words = word_tokenize(hadith)
        new_list = []
        for word in words:
            stem = ArListem.light_stem(word)
            stem = ArListem.get_stem()
            new_list.append(stem)

        hadith_sentence_with_stemming = " ".join(new_list)
        hadiths_without_stop_words_and_with_normalization_and_with_stemming.append(hadith_sentence_with_stemming)
        
    return hadiths_without_stop_words_and_with_normalization_and_with_stemming


# In[8]:


def lemmatization(text):
    """
        This function for lemma Arabic words by API, and it getting best result of the previous functions
        return a string dictinary like exactly '{"result":["امر","ب","أخذ","ما","نهى","ه","انتهى"]}'
    """
    import http.client
    conn = http.client.HTTPSConnection("farasa-api.qcri.org")
    hadith_dict = {}
    list_pyload_input = []
    list_pyload_out = []
    length = len(text)
    for h in text[:length]:
        q = '{"text":'+'"{}"'.format(h)+'}'
        list_pyload_input.append(q)
    headers = { 'content-type': "application/json", 'cache-control': "no-cache", }
    for h in list_pyload_input:
        conn.request("POST", "/msa/webapi/lemma", h.encode('utf-8'), headers)
        res = conn.getresponse()
        data = res.read()
        list_pyload_out.append(data.decode("utf-8"))
        final_result = clearReg(list_pyload_out)     # call clearReg for clean the text
    return final_result


# 1. Stemming : algorithms work by cutting off the end or the beginning of the word, taking into account a list of common prefixes and suffixes that can be found in an inflected word.
# 2. Lemmatization : takes into consideration the morphological analysis of the words.
# Lemmatization is typically seen as much more informative than simple stemming, because stem may not be an actual word whereas lemma is an actual language word.
# 
# Upove some functions for like:
# stemming_1 by ISRIStemmer from NLTK.
# stemming_2 by Tashaphyne is an Arabic light stemmer(removing prefixes and suffixes) and give all possible segmentations.
# lemmatization by Farasa API
# By the experimental: lemmatization by Farasa have a good results.

# In[9]:


data_1 = pd.read_csv(os.path.join(root, 'Sahih Bukhari Without_Tashkel.csv'))
data_1.head()


# In[10]:


all_hadiths_1 = []

for hadith in data_1['Sahih Bukhari Without_Tashkel']:
    all_hadiths_1.append(hadith)


# In[11]:


get_ipython().run_cell_magic('time', '', "\n# Maliks Muwatta\ncleared_Hadith_1 = []           # Removing stopwords\ncleared_Hadith_1_2 = []         # Normalization\ncleared_Hadith_1_2_3 = []       # Lematization\n\nfor hadith in all_hadiths_1:\n    cleared_Hadith_1.append(removing_ar_stopwords(hadith))         # Removing stopwords\nfor hadith in cleared_Hadith_1:\n    cleared_Hadith_1_2.append(normalize(hadith))                   # Normalization\ncleared_Hadith_1_2_3 = lemmatization(cleared_Hadith_1_2)           # Lematization\n\nprint('The size of data:')\nlen(cleared_Hadith_1), len(cleared_Hadith_1_2), len(cleared_Hadith_1_2_3)")


# In[ ]:


cleared_Hadith_1_2_3[:1]


# In[ ]:


Maliks_Muwatta_preprosessing_1 = pd.DataFrame(cleared_Hadith_1_2_3, columns=['Maliks_Muwatta_Preprosessing_Cleaned'])
Maliks_Muwatta_preprosessing_1.head()


# In[ ]:


import nltk
nltk.download('punkt')


# In[ ]:




