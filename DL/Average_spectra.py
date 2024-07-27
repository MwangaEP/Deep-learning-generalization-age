#%% 
# Import modules
import numpy as np 
import pandas as pd
from collections import Counter 

import matplotlib.pyplot as plt # for making plots
import seaborn as sns

sns.set(context = "paper",
        style = "whitegrid",
        palette = "deep",
        font_scale = 2.0,
        color_codes = True,
        rc=None)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# Loading dataset  
# Upload An. funestus train data for model training

train_data = pd.read_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\set_to_train_an_fun_new.csv")
print(train_data.head())

print(train_data.shape)

# Checking class distribution in the data
print(Counter(train_data["Cat3"]))

# drops columns of no interest
train_data = train_data.drop(['Unnamed: 0'], axis = 1)
train_data.head(10)


#%%

# rename age from string to real numbers

Age = []

for row in train_data['Cat3']:
    if row == '01D':
        Age.append(1)
    
    elif row == '02D':
        Age.append(2)
    
    elif row == '03D':
        Age.append(3)

    elif row == '04D':
        Age.append(4)

    elif row == '05D':
        Age.append(5)

    elif row == '06D':
        Age.append(6)

    elif row == '07D':
        Age.append(7)

    elif row == '08D':
        Age.append(8)

    elif row == '09D':
        Age.append(9)

    elif row == '10D':
        Age.append(10)

    elif row == '11D':
        Age.append(11)

    elif row == '12D':
        Age.append(12)

    elif row == '13D':
        Age.append(13)

    elif row == '14D':
        Age.append(14)

    elif row == '15D':
        Age.append(15)

    else:
        Age.append(16)

print(Age)

train_data['Age'] = Age

# drop the column with age as a string and keep the age in intergers

train_data = train_data.drop(['Cat3'], axis = 1) 
train_data.head(5)

#%%

# Renaming the age group into two age classes

Age_group = []

for row in train_data['Age']:
    if row <= 9:
        Age_group.append('1-9')

    else:
        Age_group.append('10-17')

print(Age_group)

train_data['Age_group'] = Age_group

# drop the column with Chronological Age and keep the age structure

train_data = train_data.drop(['Age'], axis = 1) 
train_data.head(5)

#%%
# calculate the mean of each class to plot the average spectra 

young = train_data.loc[train_data['Age_group'] == '1-9']
young = pd.DataFrame(young.iloc[:,:-1].mean().T).reset_index()
# young = hum.reset_index()
young.rename(columns = {'index':'wavenumber', 0:'absorbance'}, inplace = True)
# young.to_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\young_age_average.csv", index = False)

old = train_data.loc[train_data['Age_group'] == '10-17']
old = pd.DataFrame(old.iloc[:,:-1].mean().T).reset_index()
old.rename(columns = {'index':'wavenumber', 0:'absorbance'}, inplace = True)
# old.to_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\old_age_average.csv", index = False)

#%%
sns.set(context = 'paper',
        style = 'white',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))
plt.figure(figsize = (8, 4))

plt.plot(pd.to_numeric(young['wavenumber']).sort_values(ascending = False), young['absorbance'])
plt.plot(pd.to_numeric(old['wavenumber']).sort_values(ascending = False), old['absorbance'])
plt.legend(['1-9', '10-17'])
plt.xlabel("Wavenumbers / cm-1", weight = 'bold')
plt.ylabel("Absorbance", weight = 'bold')
plt.xlim(4000, 500)
plt.savefig("Averaged_graph", dpi = 500, bbox_inches="tight")
# %%
