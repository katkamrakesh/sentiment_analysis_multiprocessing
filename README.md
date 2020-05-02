## Sentiment Analysis of tweets in multiprocessing system.

Course Description:
- Course: Advance Operating System, Fall 2019
- Program: Master of Science
- Major: Computer Science
- Department of Computer Science and Mathematics
- University of Central Missouri

Goal: The objectivity of this project is to perform Sentiment Analysis of twitter data in serial and multiprocessing envirnment, comparing the processing time and system resource consumption on different systems/servers configurations. The program is executed on local system, university servers and on cloud services. The best performance is achieved for data around 1 gigabyte on cloud servers with configuration of 16 core processor and 64 gb of memory.

**TweetsPull.py script**
This script uses pythons tweepy library to connect to the twitter servers and fetch the required tweets. It uses authentication keys for connecting the twitter securely. It can fetch single hashtag or multiple hashtag in serial or multiprocessing fashion. For multiprocessing fashion, implemented custom [round robin](https://en.wikipedia.org/wiki/Round-robin_scheduling) algorithm (one of the scheduling algorithms used in operating system), to fetch 100 tweets per round of given hashtag. A priorty key and tweet numbers are set for each tag, and based on these values, the round robin algorithm will fetch the tweets. The out of this script is individual text file for each tag, and a graph plot visualising the time for serial and parallel processes.

**tweetsAnalysis.py script**
This script perform sentiment analysis of the twitter data collected from above script. Performed text cleaning, spceial character removel, url stripping and standard NLP techinques such as sentence segmentation, word tokenization, text lemmatization, and stop words identification. Used textBlob functionality for finding the polarity and subjectivity of the given tweets. The analysis can be execute in serial or paralley for all the hashtags. This script outputs polarity and subjectivity graph for individual tags, and comparison plot for serial vs multiprocessor.

***Note:***
To execute the script, put in twitter security token pair in TweetsPull.py file, and run via command line or any python editor.
