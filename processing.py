import os
import pandas as pd
import ipynb
from ipynb.fs.full.text_preprocess import preprocess
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sb
from afinn import Afinn
from ipynb.fs.full.text_preprocess import preprocess_stopwords
from sklearn import preprocessing
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from nltk.probability import FreqDist

# read data
os.chdir('C:/Users/Yufan.Wang/Documents')
merged_notes = pd.read_csv('merged_notes.csv')
merged_notes = merged_notes.drop(merged_notes.columns[0], axis=1)

# missing value
merged_notes.isnull().sum()
#missing value only in the enrolled patients, 117 out of 133 enrolled pt
merged_notes = merged_notes.fillna('no_records')

# sentiment analysis with Afinn and Textblob 
# NOTE: before processing the text; also tried after; make more sence before processing; 
af=Afinn()
merged_notes['sentiment1'] = merged_notes['ptnote'].apply(lambda x: af.score(x))
merged_notes['sentiment2'] = merged_notes['ptnote'].apply(lambda x: TextBlob(x).sentiment.polarity)

# normalise the result of afinn; result of textblob is already normalised
x_array = np.array(merged_notes['sentiment1'])
normalised=preprocessing.normalize([x_array])
normalised=normalised.reshape(-1,1)
merged_notes['sentiment_normalised'] = pd.DataFrame(normalised)

# run the process function on each note
merged_notes['processed'] = merged_notes['ptnote'].apply(lambda x: preprocess(x))

# count words in each notes
merged_notes['wordcount'] = merged_notes['processed'].apply(lambda x: len(x))

# got the stat info for a numerical column
merged_notes['sentiment2'].describe()

# find notes by condition
merged_notes.loc[merged_notes['sentiment_normalised'] < -0.05,'ptnote']

# find notes by index
merged_notes.loc[1017,'ptnote']

# compare the result of afin and textblob
plt.hist(sentiment1,bins = 20, alpha=0.5, label='Afinn')
plt.hist(sentiment2, bins=20, alpha=0.5, label = 'TextBlob')
plt.xlim(-0.5,0.5)
plt.legend(loc='upper left')

# create word cloud

words1 = merged_notes.loc[merged_notes['CurrentStatus']=='Ceased', 'processed']
allwords1 = []
for wordlist in words1:
    allwords1 += wordlist
words2 = merged_notes.loc[merged_notes['CurrentStatus']=='Complete', 'processed']
allwords2 = []
for wordlist in words2:
    allwords2 += wordlist   
    
STOPWORDS.update(["call'",'nurse',"message'","left'","advised'","member'"])

wordcloud1 = WordCloud(stopwords=STOPWORDS,max_words = 50, \
                      background_color = 'white').generate(str(allwords1))
wordcloud2 = WordCloud(stopwords=STOPWORDS,max_words = 50, \
                      background_color = 'white').generate(str(allwords2))

plt.imshow(wordcloud2)
plt.axis("off")

mostcommon1 = FreqDist(allwords1).most_common(50)
mostcommon2 = FreqDist(allwords2).most_common(25)
x1, y1 = zip(*mostcommon1)
x2, y2 = zip(*mostcommon2) # to make plt

