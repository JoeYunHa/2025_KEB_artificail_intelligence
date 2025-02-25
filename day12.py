# Assingment
# v0.9) v0.8 파일의 결측치 값을 산술평균으로 채워 넣는 방법을 적용하시오.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame(
    {
        'A':[1,2,np.nan,4],
        'B':[np.nan,12,3,4],
        'C':[1,2,3,4]
    }
)
print(df)

# using option 3 -> median
A_median = df['A'].median()
B_median = df['B'].median()
df['A'].fillna(A_median, inplace=True)
df['B'].fillna(B_median, inplace=True)

print("using fillna()")
print(df)

# using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['A','B']] = imputer.fit_transform(df[['A','B']])

print(df)
