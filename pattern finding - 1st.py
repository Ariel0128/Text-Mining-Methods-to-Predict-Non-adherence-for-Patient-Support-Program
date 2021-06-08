# create dummy columns

import pandas as pd

df['Gender'] = [0 if x == 'Male' else 1 for x in df['Gender']]
dummy_age = pd.get_dummies(df['AgeRange'], prefix='AGE', drop_first=True)
dummy_method = pd.get_dummies(df['EnrolmentMethod'], prefix='Method', drop_first=True)
dummy_journey = pd.get_dummies(df['JourneyName'], prefix='Journey', drop_first=True)

df.drop(columns=['AgeRange','EnrolmentMethod','JourneyName'], axis=1, inplace=True)

# fisher's test and chi-square test

from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact

df_noenroll=df1[df1['CurrentStatus']!='Enrolled']
contigency= pd.crosstab(df_noenroll['Journey_Pre Script Journey'], df_noenroll['CurrentStatus']) 
contigency
#p value = 0.9999999999990286

oddsratio, pvalue = fisher_exact(contigency,'greater')
pvalue

c, p, dof, expected = chi2_contingency(contigency) 
p

# write contingency 
contigency = pd.DataFrame({'EnrolmentMethod': ['granderfather','others'],
                   'ceased': [1,210],
                   'complete': [36, 917]})
contigency=contigency.set_index(contigency.columns[0])
#0.999463155640714

# heat map
plt.figure(figsize=(12,8)) 
sb.heatmap(contigency, annot=True, cmap="YlGnBu")

