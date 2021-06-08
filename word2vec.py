import gensim

os.chdir('C:/Users/Yufan.Wang/Documents')
df = pd.read_csv('merged_features.csv')
df = df.drop(df.columns[0], axis=1)

# using gensim function to do preprocessing
df['text_clean']=df['ptnote'].apply(lambda x: gensim.utils.simple_preprocess(x))

# build w2v model based on our own notes; vector_size is tunable
w2v_model = gensim.models.Word2Vec(df['text_clean'],
                                   vector_size=800,
                                   window=5,
                                   min_count=2)

# check synonyms 
w2v_model.wv.most_similar('survey',topn=20)

# Replace the words in each text message with the learned word vector
words = set(w2v_model.wv.index_to_key)
train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in df['processed']])

# Average the word vectors for each sentence (and assign a vector of zeros if the model
# did not learn any of the words in the text message during training
train_vect_avg = []
for v in train_vect:
    if v.size:
        train_vect_avg.append(v.mean(axis=0))
    else:
        train_vect_avg.append(np.zeros(800, dtype=float))
        
len(train_vect_avg)
df2 = pd.DataFrame(list(map(np.ravel, train_vect_avg)))

#test the feasibility of converting arrays of array into dataframe and send to a model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf_model = rf.fit(df3, df3['CurrentStatus'])


