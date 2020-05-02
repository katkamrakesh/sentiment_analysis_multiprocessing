#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:19:05 2019

@author: rakeshkatkam

Course: Advance Operating System, Fall 2019
Program: Master of Science
Major: Computer Science
Department of Computer Science and Mathematics
University of Central Missouri
"""

import sys
import os
import tweepy #Library to fetch tweets from twitter
import collections
from os import path
import itertools as it
import timeit
import multiprocessing as mp

API_KEY = ''
API_SECRET = ''

#==============================AUTH SET=======================================#
# Replace the API_KEY and API_SECRET with your application's key and secret.
auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

#Set path to directory of the script
dir_path = os.path.dirname(os.path.abspath(__file__))

#==============================Fetch tweets in batch of 100=======================================#

def fetchTweets(qList):
    fName = dir_path+'/'+qList[0]+".txt"
    print("\n")
    print("Downloading tweets for #{0}".format(qList[0]))
    searchQuery = "#"+qList[0]
    tweetsPerQry = 100  # this is the max the API permits
    max_id = qList[1]
    if(os.path.exists(fName)):        
        f = open(fName, 'a+')
    else:
        f = open(fName, 'w')
    try:
        if (max_id <= 0):
            new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
        else:
            new_tweets = api.search(q=searchQuery, count=tweetsPerQry,max_id=str(max_id - 1))
        if not new_tweets:
            print("No more tweets found for #{0}".format(qList[0]))
            return 0      
        for tweet in new_tweets:
            f.write(tweet.text + '\n')
        max_id = new_tweets[-1].id
    except tweepy.TweepError as e:
        print("some error : " + str(e))
    return max_id


#==============================Round Robin=======================================#

## Fetch the tweets from multiple hashtags in round robin fashion, with 100 tweets per round
def roundRobin(hastTagDict):    
    sortedDict = collections.OrderedDict(sorted(hastTagDict.items()))
    dLength = len(sortedDict)
    nCount = 0
    while(nCount < dLength):
        for i,(key, value) in enumerate(sortedDict.items()):
            if value[1] > 0:
                val = value[1] - 100
                if(val == 0):
                    nCount = nCount + 1                
                max_id = fetchTweets([value[0],value[2]])
                tCount = value[3]+100
                print ("Downloaded {0} tweets, Saved to {1}.txt".format(tCount,value[0]))
                print("\n")
                if max_id == 0:
                    val = 0
                    nCount = nCount + 1
                tempDict = {key : [value[0],val,max_id,tCount]}
                sortedDict.update(tempDict)                                   


##Fetch the tweets in serial process
def serialProc(hastTagDict):
    print("========Serial Processing Starts=========== \n")
    roundRobin(hastTagDict)
    print("========Serial Processing End=========== \n")


##Fetch the tweets in parallel process
def multiProc(hastTagDict,numProcessors):
    d1 = {k: hastTagDict[k] for k in filter(lambda x: x <= 2, hastTagDict.keys())}
    d2 = {k: hastTagDict[k] for k in filter(lambda x: x > 2, hastTagDict.keys())}
    fileList = [d1,d2]
    pool = mp.Pool(processes=numProcessors)
    results = [pool.apply_async(roundRobin, args=(x,)) for x in fileList]
    output = [p.get() for p in results]
    print(output)


hastTagDict = {2:["CoronaVirus",10000,-1,0],1:["covid19",10000,-1,0]}

benchmarks = []

benchmarks.append(timeit.Timer('serialProc(hastTagDict)',
            'from __main__ import serialProc, hastTagDict').timeit(number=1))

benchmarks.append(timeit.Timer('multiProc(hastTagDict, 2)',
            'from __main__ import multiProc, hastTagDict').timeit(number=1))

benchmarks.append(timeit.Timer('multiProc(hastTagDict, 3)',
            'from __main__ import multiProc, hastTagDict').timeit(number=1))


##Plot the comparison between serial vs parallel process
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

    plt.xlabel('time in seconds for n=%s', fontsize=14)
    plt.ylabel('number of processes', fontsize=14)
    t = plt.title('Serial vs. Multiprocessing', fontsize=18)
    plt.ylim([-1,len(benchmarks)+0.5])
    plt.xlim([0,max(benchmarks)*1.1])
    plt.vlines(benchmarks[0], -1, len(benchmarks)+0.5, linestyles='dashed')
    plt.grid()

    plt.show()
    
plot_results()