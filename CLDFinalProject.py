
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
get_ipython().magic(u'matplotlib inline')


# In[2]:

df = pd.read_csv('NHES_PFI.csv')


# In[3]:

df.head()


# In[4]:

df = df.drop(['BASMID', 'RCVDATE'], axis = 1)


# In[5]:

df = df[df.QTYPE != 1]


# In[6]:

df.head(n=10)


# In[7]:

df=df.drop(['QTYPE', 'HOMESCHLX'], axis=1)


# In[8]:

pd.set_option('display.max_columns', None)
df.describe()


# In[9]:

df_nonwhite = df[df.CWHITE != 1]


# In[10]:

df_allwhite = df[df.CWHITE == 1]
df_allwhite = df_allwhite[df_allwhite['CHISPAN'] != 1]


# In[11]:

df_whiteCLD = df_allwhite[df.CSPEAKX > 2]


# In[12]:

df_white = df_allwhite[df_allwhite.CSPEAKX < 3]


# In[13]:

df_cld = df_nonwhite.append(df_whiteCLD)


# In[ ]:




# In[14]:

df_cld['HDIEP'].value_counts()


# In[15]:

df_cld['HDIEP'].value_counts(normalize=True)


# In[16]:

df_white['HDIEP'].value_counts()


# In[17]:

df_white['HDIEP'].value_counts(normalize=True)


# In[18]:

print 'HDIEP' in df_white


# In[19]:

df_white = df_white.rename(index=str, columns={'HDIEP': 'IEP'})


# In[20]:

df_cld = df_cld.rename(index=str, columns={'HDIEP': 'IEP'})


# In[21]:

df_white_iep = df_white[df_white['IEP'] == 1]


# In[22]:

df_cld_iep = df_cld[df_cld.IEP == 1]


# In[23]:

df_white_iep.describe()


# In[24]:

df_cld_iep.describe()


# In[25]:

cld_counts = df_cld_iep.apply(pd.value_counts, normalize=True)


# In[26]:

cld_counts.head(n=10)


# In[27]:

white_counts = df_white_iep.apply(pd.value_counts, normalize=True)


# In[28]:

white_counts.head(n=10)


# In[29]:

df = df.rename(index=str, columns={'HDIEP': 'IEP'})
df_iep = df[df['IEP'] == 1]


# In[30]:

counts = df_iep.apply(pd.value_counts, normalize=True)


# In[31]:

df_fb_iep = df_iep[df_iep['CPLCBRTH'] > 1]
df_hispanic_iep = df_iep[df_iep['CHISPAN'] == 1]
df_amind_iep = df_iep[df_iep['CAMIND'] == 1]
df_asian_iep = df_iep[df_iep['CASIAN'] == 1]
df_black_iep = df_iep[df_iep['CBLACK'] == 1]
df_pi_iep = df_iep[df_iep['CPACI'] == 1]
df_spanish_iep = df_iep[df_iep['CSPEAKX'] == 3]
df_bilingualspanish_iep = df_iep[df_iep['CSPEAKX']==5]
df_spanish_iep = df_bilingualspanish_iep.append(df_spanish_iep)
df_otherlang_iep = df_iep[df_iep['CSPEAKX'] == 4]
df_otherbilingual_iep = df_iep[df_iep['CSPEAKX'] == 6]
df_otherlang_iep = df_otherbilingual_iep.append(df_otherlang_iep)
df_efl_iep = df_iep[df_iep['CENGLPRG'] == 1]


# In[32]:

fb_counts = df_iep.apply(pd.value_counts, normalize=True)
hispanic_counts = df_hispanic_iep.apply(pd.value_counts, normalize=True)
amind_counts = df_amind_iep.apply(pd.value_counts, normalize=True)
asian_counts = df_asian_iep.apply(pd.value_counts, normalize=True)
black_counts = df_black_iep.apply(pd.value_counts, normalize=True)
pi_counts = df_pi_iep.apply(pd.value_counts, normalize=True)
spanish_counts = df_spanish_iep.apply(pd.value_counts, normalize=True)
otherlang_counts = df_otherlang_iep.apply(pd.value_counts, normalize=True)
efl_counts = df_efl_iep.apply(pd.value_counts, normalize=True)


# In[ ]:




# In[33]:

disabilities = counts.ix[4, 'HDLEARNX':'HDOTHERX']
disabilities


# In[34]:

disabilities_cld = cld_counts.ix[4, 'HDLEARNX':'HDOTHERX']
disabilities_white = white_counts.ix[4, 'HDLEARNX':'HDOTHERX']
disabilities_fb = fb_counts.ix[4, 'HDLEARNX':'HDOTHERX']
disabilities_hispanic = hispanic_counts.ix[4, 'HDLEARNX':'HDOTHERX']
disabilities_amind = amind_counts.ix[4, 'HDLEARNX':'HDOTHERX']
disabilities_asian = asian_counts.ix[3, 'HDLEARNX':'HDOTHERX']
disabilities_black = black_counts.ix[4, 'HDLEARNX':'HDOTHERX']
disabilities_pi = pi_counts.ix[3, 'HDLEARNX':'HDOTHERX']
disabilities_spanish = spanish_counts.ix[3, 'HDLEARNX':'HDOTHERX']
disabilities_otherlang = otherlang_counts.ix[3, 'HDLEARNX':'HDOTHERX']
disabilities_efl = efl_counts.ix[3, 'HDLEARNX':'HDOTHERX']
disabilities_efl


# In[35]:

compare_disabilities = pd.concat([disabilities, disabilities_cld, disabilities_white, disabilities_fb, disabilities_hispanic, disabilities_amind, disabilities_asian, disabilities_black, disabilities_pi, disabilities_spanish, disabilities_otherlang, disabilities_efl], axis =1).T.reset_index()


# In[36]:

compare_disabilities


# In[37]:

compare_disabilities = compare_disabilities.rename(index=str, columns={'index': 'Students', 'HDLEARNX': 'SLD', 'HDINTDIS': 'ID', 'HDSPEECHX': 'SLI', 'HDDISTRBX': 'EBD', 'HDDEAFIMX': 'Deaf', 'HDBLINDX': 'VI', 'HDORTHOX': 'Ortho', 'HDAUTISMX': 'Autism', 'HDPDDX': 'PDD', 'HDADDX': 'ADD', 'HDDELAYX': 'Delay', 'HDTRBRAIN': 'TBI', 'HDOTHERX': 'OHI'})


# In[38]:

compare_disabilities


# In[39]:

ad = compare_disabilities.index[0]
cldd = compare_disabilities.index[1]
wd = compare_disabilities.index[2]
fbd = compare_disabilities.index[3]
hd = compare_disabilities.index[4]
aid = compare_disabilities.index[5]
asd = compare_disabilities.index[6]
bd = compare_disabilities.index[7]
pid = compare_disabilities.index[8]
sd = compare_disabilities.index[9]
old = compare_disabilities.index[10]
efld = compare_disabilities.index[11]


# In[40]:

compare_disabilities = compare_disabilities.rename(index={ad: 'All', cldd: 'CLD', wd: 'Not_CLD', fbd: 'ForeignBorn', hd: 'Hispanic', aid: 'AmericanIndian', asd: 'Asian', bd: 'Black', pid: 'PacificIslander', sd: 'Spanish', old: 'OtherLanguage', efld: 'InEFL'})


# In[82]:

compare_disabilities.to_csv('disabilities.csv')


# In[41]:

compare_disabilities

compare_disabilities.to_csv
# In[42]:

sped_ratings = counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_cld = cld_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_white = white_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_fb = fb_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_hispanic = hispanic_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_amind = amind_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_asian = asian_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_black = black_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_pi = pi_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_spanish = spanish_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_otherlang = otherlang_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings_efl = efl_counts.ix[4:6, 'HDCOMMUX':'HDCOMMITX']
sped_ratings.loc['Satisfied'] = sped_ratings.iloc[0] + sped_ratings.iloc[1]
sped_ratings_cld.loc['Satisfied'] = sped_ratings_cld.iloc[0] + sped_ratings_cld.iloc[1]
sped_ratings_white.loc['Satisfied'] = sped_ratings_white.iloc[0] + sped_ratings_white.iloc[1]
sped_ratings_fb.loc['Satisfied'] = sped_ratings_fb.iloc[0] + sped_ratings_fb.iloc[1]
sped_ratings_hispanic.loc['Satisfied'] = sped_ratings_hispanic.iloc[0] + sped_ratings_hispanic.iloc[1]
sped_ratings_amind.loc['Satisfied'] = sped_ratings_amind.iloc[0] + sped_ratings_amind.iloc[1]
sped_ratings_asian.loc['Satisfied'] = sped_ratings_asian.iloc[0] + sped_ratings_asian.iloc[1]
sped_ratings_black.loc['Satisfied'] = sped_ratings_black.iloc[0] + sped_ratings_black.iloc[1]
sped_ratings_pi.loc['Satisfied'] = sped_ratings_pi.iloc[0] + sped_ratings_pi.iloc[1]
sped_ratings_spanish.loc['Satisfied'] = sped_ratings_spanish.iloc[0] + sped_ratings_spanish.iloc[1]
sped_ratings_otherlang.loc['Satisfied'] = sped_ratings_otherlang.iloc[0] + sped_ratings_otherlang.iloc[1]
sped_ratings_efl.loc['Satisfied'] = sped_ratings_efl.iloc[0] + sped_ratings_efl.iloc[1]
sped_ratings = sped_ratings.ix[2, ]
sped_ratings_cld = sped_ratings_cld.ix[2, ]
sped_ratings_white = sped_ratings_white.ix[2, ]
sped_ratings_fb = sped_ratings_fb.ix[2, ]
sped_ratings_hispanic = sped_ratings_hispanic.ix[2, ]
sped_ratings_amind = sped_ratings_amind.ix[2, ]
sped_ratings_asian = sped_ratings_asian.ix[2, ]
sped_ratings_black = sped_ratings_black.ix[2, ]
sped_ratings_pi = sped_ratings_pi.ix[2, ]
sped_ratings_spanish = sped_ratings_spanish.ix[2, ]
sped_ratings_otherlang = sped_ratings_otherlang.ix[2, ]
sped_ratings_efl = sped_ratings_efl.ix[2, ]


# In[ ]:




# In[43]:

sped_ratings


# In[ ]:




# In[44]:

compare_sped_ratings = pd.concat([sped_ratings, sped_ratings_cld, sped_ratings_white, sped_ratings_fb, sped_ratings_hispanic, sped_ratings_amind, sped_ratings_asian, sped_ratings_black, sped_ratings_pi, sped_ratings_spanish, sped_ratings_otherlang, sped_ratings_efl], axis =1).T.reset_index()
compare_sped_ratings


# In[45]:

asped = compare_sped_ratings.index[0]
cldsped = compare_sped_ratings.index[1]
wsped = compare_sped_ratings.index[2]
fbsped = compare_sped_ratings.index[3]
hsped = compare_sped_ratings.index[4]
aisped = compare_sped_ratings.index[5]
assped = compare_sped_ratings.index[6]
bsped = compare_sped_ratings.index[7]
pisped = compare_sped_ratings.index[8]
ssped = compare_sped_ratings.index[9]
olsped = compare_sped_ratings.index[10]
eflsped = compare_sped_ratings.index[11]

compare_sped_ratings = compare_sped_ratings.rename(index={asped: 'All', cldsped: 'CLD', wsped: 'Not_CLD'})
compare_sped_ratings


# In[83]:

compare_sped_ratings.to_csv('spedratings.csv')


# In[59]:

school_ratings = counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_cld = cld_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_white = white_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_fb = fb_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_hispanic = hispanic_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_amind = amind_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_asian = asian_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_black = black_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_pi = pi_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_spanish = spanish_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_otherlang = otherlang_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']
school_ratings_efl = efl_counts.ix[4:6, 'FSSPPERF':'FCSUPPRT']


# In[60]:

school_ratings.loc['Satisfied'] = school_ratings.iloc[0] + school_ratings.iloc[1]
school_ratings_cld.loc['Satisfied'] = school_ratings_cld.iloc[0] + school_ratings_cld.iloc[1]
school_ratings_white.loc['Satisfied'] = school_ratings_white.iloc[0] + school_ratings_white.iloc[1]
school_ratings_fb.loc['Satisfied'] = school_ratings_fb.iloc[0] + school_ratings_fb.iloc[1]
school_ratings_hispanic.loc['Satisfied'] = school_ratings_hispanic.iloc[0] + school_ratings_hispanic.iloc[1]
school_ratings_amind.loc['Satisfied'] = school_ratings_amind.iloc[0] + school_ratings_amind.iloc[1]
school_ratings_asian.loc['Satisfied'] = school_ratings_asian.iloc[0] + school_ratings_asian.iloc[1]
school_ratings_black.loc['Satisfied'] = school_ratings_black.iloc[0] + school_ratings_black.iloc[1]
school_ratings_pi.loc['Satisfied'] = school_ratings_pi.iloc[0] + school_ratings_pi.iloc[1]
school_ratings_spanish.loc['Satisfied'] = school_ratings_spanish.iloc[0] + school_ratings_spanish.iloc[1]
school_ratings_otherlang.loc['Satisfied'] = school_ratings_otherlang.iloc[0] + school_ratings_otherlang.iloc[1]
school_ratings_efl.loc['Satisfied'] = school_ratings_efl.iloc[0] + school_ratings_efl.iloc[1]

school_ratings = school_ratings.ix[2, ]
school_ratings_cld = school_ratings_cld.ix[2, ]
school_ratings_white = school_ratings_white.ix[2, ]
school_ratings_fb = school_ratings_fb.ix[2, ]
school_ratings_hispanic = school_ratings_hispanic.ix[2, ]
school_ratings_amind = school_ratings_amind.ix[2, ]
school_ratings_asian = school_ratings_asian.ix[2, ]
school_ratings_black = school_ratings_black.ix[2, ]
school_ratings_pi = school_ratings_pi.ix[2, ]
school_ratings_spanish = school_ratings_spanish.ix[2, ]
school_ratings_otherlang = school_ratings_otherlang.ix[2, ]
school_ratings_efl = school_ratings_efl.ix[2, ]

compare_school_ratings = pd.concat([school_ratings, school_ratings_cld, school_ratings_white, school_ratings_fb, school_ratings_hispanic, school_ratings_amind, school_ratings_asian, school_ratings_black, school_ratings_pi, school_ratings_spanish, school_ratings_otherlang, school_ratings_efl], axis =1).T.reset_index()
asch = compare_school_ratings.index[0]
cldsch = compare_school_ratings.index[1]
wsch = compare_school_ratings.index[2]

compare_school_ratings = compare_school_ratings.rename(index={asch: 'All', cldsch: 'CLD', wsch: 'Not_CLD'})
compare_school_ratings


# In[86]:

compare_school_ratings.to_csv('schoolratings.csv')


# In[48]:

invol = counts.ix[4, 'FSSPORTX': 'FSCOUNSLR']
invol_cld = cld_counts.ix[4, 'FSSPORTX':'FSCOUNSLR']
invol_white = white_counts.ix[4, 'FSSPORTX':'FSCOUNSLR']
invol_fb = fb_counts.ix[4, 'FSSPORTX':'FSCOUNSLR']
invol_hispanic = hispanic_counts.ix[4, 'FSSPORTX':'FSCOUNSLR']
invol_amind = amind_counts.ix[4, 'FSSPORTX':'FSCOUNSLR']
invol_asian = asian_counts.ix[3, 'FSSPORTX':'FSCOUNSLR']
invol_black = black_counts.ix[4, 'FSSPORTX':'FSCOUNSLR']
invol_pi = pi_counts.ix[3, 'FSSPORTX':'FSCOUNSLR']
invol_spanish = spanish_counts.ix[3, 'FSSPORTX':'FSCOUNSLR']
invol_otherlang = otherlang_counts.ix[3, 'FSSPORTX':'FSCOUNSLR']
invol_efl = efl_counts.ix[3, 'FSSPORTX':'FSCOUNSLR']


# In[49]:

compare_invol = pd.concat([invol, invol_cld, invol_white, invol_fb, invol_hispanic, invol_amind, invol_asian, invol_black, invol_pi, invol_spanish, invol_otherlang, invol_efl], axis =1).T.reset_index()
compare_invol


# In[84]:

compare_invol.to_csv('involvement.csv')


# In[61]:

comm = counts.ix[4, 'FSNOTESX': 'FSPHONCHX']
comm_cld = cld_counts.ix[4, 'FSNOTESX': 'FSPHONCHX']
comm_white = white_counts.ix[4, 'FSNOTESX': 'FSPHONCHX']
comm_fb = fb_counts.ix[4, 'FSNOTESX': 'FSPHONCHX']
comm_hispanic = hispanic_counts.ix[4, 'FSNOTESX': 'FSPHONCHX']
comm_amind = amind_counts.ix[4, 'FSNOTESX': 'FSPHONCHX']
comm_asian = asian_counts.ix[3, 'FSNOTESX': 'FSPHONCHX']
comm_black = black_counts.ix[4, 'FSNOTESX': 'FSPHONCHX']
comm_pi = pi_counts.ix[3, 'FSNOTESX': 'FSPHONCHX']
comm_spanish = spanish_counts.ix[3, 'FSNOTESX': 'FSPHONCHX']
comm_otherlang = otherlang_counts.ix[3, 'FSNOTESX': 'FSPHONCHX']
comm_efl = efl_counts.ix[3, 'FSNOTESX': 'FSPHONCHX']


# In[62]:

compare_comm = pd.concat([comm, comm_cld, comm_white, comm_fb, comm_hispanic, comm_amind, comm_asian, comm_black, comm_pi, comm_spanish, comm_otherlang, comm_efl], axis =1).T.reset_index()
compare_comm


# In[85]:

compare_comm.to_csv('communication.csv')


# In[50]:

df_iep['nonenglish'] = [1 if x > 2 else 0 for x in df_iep['CSPEAKX']]


# In[51]:

df_iep = df_iep.drop(['PATH', 'ALLGRADEX'], axis = 1)


# In[52]:

df_iep = df_iep.astype(float)


# In[53]:

df_iep.convert_objects(convert_numeric=True)
df_iep.convert_objects(convert_numeric=True).dtypes


# In[63]:


pca = PCA(svd_solver='full')
pca.fit(df_iep)
T = pca.transform(df_iep)


# In[64]:

import assignment2_helper as helper
scaleFeatures = True


# In[65]:

ax = helper.drawVectors(T, pca.components_, df_iep.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


# In[69]:

df_spanish_iep = df_spanish_iep.drop(['PATH', 'ALLGRADEX'], axis = 1)
df_spanish_iep = df_spanish_iep.astype(float)


# In[70]:

pca = PCA(svd_solver='full')
pca.fit(df_spanish_iep)
T = pca.transform(df_spanish_iep)


# In[71]:

ax = helper.drawVectors(T, pca.components_, df_spanish_iep.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


# In[66]:

df_black_iep = df_black_iep.drop(['PATH', 'ALLGRADEX'], axis = 1)
df_black_iep = df_black_iep.astype(float)


# In[67]:

pca = PCA(svd_solver='full')
pca.fit(df_black_iep)
T = pca.transform(df_black_iep)


# In[68]:

ax = helper.drawVectors(T, pca.components_, df_black_iep.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


# In[72]:

df_asian_iep = df_asian_iep.drop(['PATH', 'ALLGRADEX'], axis = 1)
df_asian_iep = df_asian_iep.astype(float)


# In[73]:

pca = PCA(svd_solver='full')
pca.fit(df_asian_iep)
T = pca.transform(df_asian_iep)


# In[74]:

ax = helper.drawVectors(T, pca.components_, df_asian_iep.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


# In[75]:

df_efl_iep = df_efl_iep.drop(['PATH', 'ALLGRADEX'], axis = 1)
df_efl_iep = df_efl_iep.astype(float)


# In[76]:

pca = PCA(svd_solver='full')
pca.fit(df_efl_iep)
T = pca.transform(df_efl_iep)


# In[77]:

ax = helper.drawVectors(T, pca.components_, df_efl_iep.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


# In[78]:

df_fb_iep = df_fb_iep.drop(['PATH', 'ALLGRADEX'], axis = 1)
df_fb_iep = df_fb_iep.astype(float)


# In[79]:

pca = PCA(svd_solver='full')
pca.fit(df_fb_iep)
T = pca.transform(df_fb_iep)


# In[80]:

ax = helper.drawVectors(T, pca.components_, df_fb_iep.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


# In[ ]:



