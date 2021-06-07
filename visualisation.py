### boxplot and scatter

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
sp = sb.stripplot(x=dates_topic5['CurrentStatus'], y=dates_topic5['DaysToEnd'], 
                   hue=dates_topic5['CurrentStatus'], data=dates_topic5, ax=ax1)
bp = sb.boxplot(x=dates_topic5['CurrentStatus'], y=dates_topic5['DaysToEnd'], 
                 hue=dates_topic5['CurrentStatus'], data=dates_topic5, palette="Set2", ax=ax2)
t = f.suptitle('Visualizing Pt words', fontsize=20)

### just scatter

facet = sb.lmplot(data=dates_topic5, x='DaysToEnd', y='DaysToStart', hue='CurrentStatus', 
                   fit_reg=False, legend=True, legend_out=True)

facet = sb.lmplot(data=df, x='duration', y='wordcount', hue='CurrentStatus', 
                   fit_reg=False, legend=True, legend_out=True)

### histogram, ordered by value

ax = newdf.loc[newdf['JourneyName']=='Plaque Psoriasis (PSO) Low Touch Journey','AgeRange'].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
ax.set_title('Credit card fraud (0 = normal, 1 = fraud)', size=20, pad=30)
ax.set_ylabel('Number of transactions', fontsize=14)
for i in ax.patches:
    ax.text(i.get_x() + 0.19, i.get_height() + 1, str(round(i.get_height(), 2)), fontsize=15)

    
### histogram, with opacity and bins

plt.hist(age_WholeProgram, alpha=0.5, label='For Whole Program')
plt.hist(ceased, alpha=0.5, label = 'For Pt Mentioned The Topic')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

plt.hist(sentiment1,bins = 20, alpha=0.5, label='Afinn')
plt.hist(sentiment2, bins=20, alpha=0.5, label = 'TextBlob')
plt.xlim(-0.5,0.5)
plt.legend(loc='upper left')

