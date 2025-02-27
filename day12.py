# Assingment
# v1.1) 나이에 따른 생존율을 고려하여 모델을 선택하고
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

titanic = sns.load_dataset('titanic')

# visualized to select model

survived_data = titanic[titanic['survived'] == 1]

dead_data = titanic[titanic['survived']==0]

survived_data['age_group'] = pd.cut(survived_data['age'], bins=[0,10,20,30,40,50,60,70,80,90,100], labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90+'])
dead_data['age_group'] = pd.cut(dead_data['age'], bins=[0,10,20,30,40,50,60,70,80,90,100], labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90+'])

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(survived_data['age_group'].value_counts().index))
width = 0.35

ax.bar(x - width/2, survived_data['age_group'].value_counts().sort_index().values, width, label='생존자', alpha=0.5, color='blue')
ax.bar(x + width/2, dead_data['age_group'].value_counts().sort_index().values, width, label='사망자', alpha=0.5, color='red')

ax.set_xticks(x)
ax.set_xticklabels(survived_data['age_group'].value_counts().index)
ax.set_xlabel('age')
ax.set_ylabel('counts')

ax.legend()

plt.show()