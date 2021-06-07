### remove missing value


# Count Vector
# Max_df=0.9 will remove words that appear in more than 90% of the reviews. 
# Min_df=25 will remove words that appear in less than 25 reviews.

vectorizer = CountVectorizer(max_df=0.9, min_df=25, max_features=5000, ngram_range=(2,2))
termfreq = vectorizer.fit_transform(df['string'].values.astype('U'))
features = vectorizer.get_feature_names()
term_matrix = pd.DataFrame(termfreq.toarray(), columns = list(features))
term_matrix.head(5)

# LDA
# produce 10 individual topics (ie. n_components).
# Each topic will consist of 10 words

lda = LatentDirichletAllocation(n_components=10, learning_method='online',
                               max_iter=500, random_state=0).fit(termfreq)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                          for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda, features, 10)

# TF-IDF
# Max_df=0.9 will remove words that appear in more than 90% of the reviews. 
# Min_df=25 will remove words that appear in less than 25 reviews.

tfidf_vector = TfidfVectorizer(max_df=0.9, min_df=25, max_features=5000, use_idf=True)
tfidf = tfidf_vector.fit_transform(df['string'])
tfidf_features = tfidf_vector.get_feature_names()
tfidf_matrix = pd.DataFrame(tfidf.toarray(), columns = list(tfidf_features))
tfidf_matrix.shape

# NMF
# Produce 20 topics, each topic has 15 key words

nmf = NMF(n_components=20, random_state=0, alpha=.1, init='nndsvd', max_iter=300).fit(tfidf)
display_topics(nmf, tfidf_features, 15)

nmf_topic_values = nmf.transform(tfidf)
df['topic'] = nmf_topic_values.argmax(axis=1)

#lda_topic_values = lda.transform(termfreq)
#merged_notes['lda_topics'] = lda_topic_values.argmax(axis=1)

#Topic remap is not meaningful here

# check the sentiment for each topic

polarity_avg = df.groupby('topic')['sentiment'].mean().plot(kind='bar')
plt.title('Average Polarity per topic')
plt.show()

# Count the topic appearance by patient

df.groupby(['pmid','topic'])['topic'].count()

#result is like this:
#pmid          topic
#TABC000009  0        1
#              1        1
#              2        1
#              3        3
#              4        4
#                      ..
#TABC001463  19       1
#TABC001464  19       1
#TABC001465  19       1
#TABC001466  19       1
#TABC001467  19       1

dfs=[]
for i in range(0,20):
    df_n = pd.DataFrame(df[df['topic']==i].groupby('pmid')['topic'].count())
    dfs.append(df_n)
newdf = pd.concat(dfs,axis=1)
newdf.columns=['topic0','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10',
               'topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19']

newdf = newdf.fillna(0)

# create the feature of notecount
notecount=pd.DataFrame(df.groupby(['pmid'])['Description'].count())
