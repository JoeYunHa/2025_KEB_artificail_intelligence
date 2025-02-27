# Assingment
# v1.1) 나이에 따른 생존율을 고려하여 모델을 선택, 학습 및 예측
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

titanic = sns.load_dataset('titanic')

# visualized to select model

survived_data = titanic[titanic['survived'] == 1].copy()

dead_data = titanic[titanic['survived']==0].copy()
# .copy() => SettingWithCopyWarning 방지

titanic['age_group'] = pd.cut(titanic['age'],
                            bins=[0,10,20,30,40,50,60,70,80,90,100],
                            labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90+'])

survived_data['age_group'] = pd.cut(survived_data['age'],
                            bins=[0,10,20,30,40,50,60,70,80,90,100],
                            labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90+'])
dead_data['age_group'] = pd.cut(dead_data['age'], bins=[0,10,20,30,40,50,60,70,80,90,100], labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90+'])

age_group_total = titanic['age_group'].value_counts().sort_index()
age_group_dead = dead_data['age_group'].value_counts().sort_index()
age_group_survived = survived_data['age_group'].value_counts().sort_index()

death_rate = (age_group_dead / age_group_total) * 100
survived_rate = (age_group_survived / age_group_total) * 100

plt.figure(figsize=(12,6))
sns.set_style("whitegrid")

# death_rate_bars = plt.bar(death_rate.index, death_rate.values, color='red')
survived_rate_bars = plt.bar(survived_rate.index, survived_rate.values, color='blue')

plt.xlabel('age', fontsize = 16)
plt.ylabel('survived_rate (%)' , fontsize=12)
plt.ylim(0,100)

for bar in survived_rate_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

