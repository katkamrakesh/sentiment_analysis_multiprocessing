#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:13:58 2019

@author: rakeshkatkam
Course: Advance Operating System, Fall 2019
Program: Master of Science
Major: Computer Science
Department of Computer Science and Mathematics
University of Central Missouri
"""

import pandas as pd
import numpy as np
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob
import multiprocessing as mp
import string
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Set path to directory of the script
dir_path = os.path.dirname(os.path.abspath(__file__))
#==============================Tweets Data Processing=======================================#

##Stripping and sanitization of tweets from special characters, emojis and other symbol
def clean_tweet(tweet): 
        text = tweet.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

##Vectorization of data matrix conversion of tweets data
def Vect(df):
    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(df.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = df.index
    return data_dtm

## Getting polarity of tweets
def calculateSentPerc(pol):
    if pol > 0:
        return 'positive'
    elif pol == 0: 
        return 'neutral'
    else: 
        return 'negative'

##Print the system information
def print_sysinfo():

    print('\nPython version  :', platform.python_version())
    print('compiler        :', platform.python_compiler())

    print('\nsystem     :', platform.system())
    print('release    :', platform.release())
    print('machine    :', platform.machine())
    print('processor  :', platform.processor())
    print('CPU count  :', mp.cpu_count())
    print('interpreter:', platform.architecture()[0])
    print('\n\n')

## Sentiment analysis of the tweets using TextBlob
def processTweets(name,flag):
    print("Tweet Processing Begins: ",flag)
    fileName = dir_path+'/'+name+".txt"
    figName = dir_path+'/'+name+"_"+flag+".pdf"
    df = pd.read_fwf(fileName)
    df.columns = ['text']
    df = pd.DataFrame(df.text.apply(clean_tweet))
    df = df.sort_index()
    
    pol = lambda x: TextBlob(x).sentiment.polarity
    sub = lambda x: TextBlob(x).sentiment.subjectivity
    
    df['polarity'] = df['text'].apply(pol)
    df['subjectivity'] = df['text'].apply(sub)
    #df
    value=(df['polarity']>0.0)
    df['color']= np.where(value==True , "#9b59b6", "#3498db")

    # Let's plot the sentiments results
    f = plt.figure()
    plt.title(name)
    plt.scatter(df.polarity, df.subjectivity, c=df.color, s=20)
    f.savefig(figName)
    
    df['sentiment'] = df['polarity'].apply(calculateSentPerc)
    
    perc = df.groupby('sentiment').count()
    
    posCount = perc.text.positive
    negCount = perc.text.negative
    neuCount = perc.text.neutral
    totCount = posCount+negCount+neuCount
    posPerc = 100*(posCount/totCount)
    negPerc = 100*(negCount/totCount)
    neuPerc = 100*(neuCount/totCount)
    print("Tweet Analysis for #"+name)
    print("Positive%: {0}, Negative%: {1}, Neutral%:{2}".format(posPerc, negPerc, neuPerc))

##Analyse the tweets in serial process
def serialProc(fileList):    
    for f in fileList:
        processTweets(f,'Serial')

##Analyse the tweets in multi-process
def multiProc(fileList,numProcessors):    
    pool = mp.Pool(processes=numProcessors)
    results = [pool.apply_async(processTweets, args=(x,'Multi',)) for x in fileList]
    output = [p.get() for p in results]
    print(output)

fileList = ['CoronaVirus']

import timeit

benchmarks = []

benchmarks.append(timeit.Timer('serialProc(fileList)',
            'from __main__ import serialProc, fileList').timeit(number=1))

benchmarks.append(timeit.Timer('multiProc(fileList, 2)',
            'from __main__ import multiProc, fileList').timeit(number=1))

benchmarks.append(timeit.Timer('multiProc(fileList, 3)',
            'from __main__ import multiProc, fileList').timeit(number=1))

##Plot the comparison between serial vs multiprocess
def plot_results():
    bar_labels = ['serial', '2', '3']

    fig = plt.figure(figsize=(10,8))

    # plot bars
    y_pos = np.arange(len(benchmarks))
    plt.yticks(y_pos, bar_labels, fontsize=16)
    bars = plt.barh(y_pos, benchmarks,
             align='center', alpha=0.4, color='b')

    # annotation and labels

    for ba,be in zip(bars, benchmarks):
        plt.text(ba.get_width() + 2, ba.get_y() + ba.get_height()/2,
                '{0:.2%}'.format(benchmarks[0]/be),
                ha='center', va='bottom', fontsize=12)

    plt.xlabel('time in seconds for n=%s' %n, fontsize=14)
    plt.ylabel('number of processes', fontsize=14)
    t = plt.title('Serial vs. Multiprocessing', fontsize=18)
    plt.ylim([-1,len(benchmarks)+0.5])
    plt.xlim([0,max(benchmarks)*1.1])
    plt.vlines(benchmarks[0], -1, len(benchmarks)+0.5, linestyles='dashed')
    plt.grid()

    plt.show()
    
print_sysinfo()
plot_results()