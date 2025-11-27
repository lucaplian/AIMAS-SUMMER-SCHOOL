# Importăm biblioteca pandas pentru a manipula datele
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.model_selection import KFold, GridSearchCV


# Citim fișierul CSV cu datele de antrenament și afișăm primele 5 rânduri
df_train = pd.read_csv('train_data.csv')
df_train.head()

# Citim fișierul CSV cu datele de test și afișăm primele 5 rânduri
df_test = pd.read_csv('test_data.csv')
df_test.head()





# Adăugăm coloane noi în setul de date de test pentru fiecare subtask
df_test['Task1'] = 0.0
df_test['Task2'] = 0.25
df_test['Task3'] = 0.0
df_test['Task4'] = 0.0
df_test['Task5'] = 1000.0


df_test['Task1'] = df_test['Square_Footage'] + df_test['Garage_Size'] + df_test['Lot_Size']
df_test['Task2'] = df_test['Garage_Size']/df_test['Total_Rooms']
df_test['Task3'] = (df_test['Solar_Exposure_Index']-df_test['Vibration_Level'])/df_test['Magnetic_Field_Strength']



'''df_train['Task1'] = df_train['Square_Footage'] + df_train['Garage_Size'] + df_train['Lot_Size']
df_train['Task2'] = df_train['Garage_Size']/df_train['Total_Rooms']
df_train['Task3'] = (df_train['Solar_Exposure_Index']-df_train['Vibration_Level'])/df_train['Magnetic_Field_Strength']'''

mean_val = df_train['Square_Footage'].mean()

for idx, rows in df_test.iterrows():
    df_test.loc[idx, 'Task4'] = abs(df_test.loc[idx, 'Square_Footage']-mean_val)


targetColumn = 'Price'
df_tes_test = df_test.corr(method='pearson')
df_tes_train = df_train.corr(method='pearson')

filtered_df_tes_train = df_tes_train[(df_tes_train[targetColumn]>=0.05) | (df_tes_train[targetColumn]<-0.05)]

for cols in df_train.columns:
    if cols == targetColumn:
        continue
    filtered_df_tes_train2 = df_tes_train[((df_tes_train[cols]>=0.7) & (df_tes_train[cols]<1)) | ((df_tes_train[cols]<-0.7) & (df_tes_train[cols]<1))]
    if len(filtered_df_tes_train2) == 1 and filtered_df_tes_train2.index.values[0] == targetColumn:
        continue

    if len(filtered_df_tes_train2) > 0:
        first = abs(df_tes_train[targetColumn][cols])
        second = abs(df_tes_train[targetColumn][filtered_df_tes_train2.index.values[0]])

        if first > second:
            df_train.drop(columns=[filtered_df_tes_train2.index.values[0]])
        else:
            df_train.drop(columns=[cols])
            
for cols in df_train.columns:
    if df_train[cols].describe()['std'] > 20:
        percentile_low = df_train[cols].sort_values(ascending=True).iloc[len(df_train[cols])//10]
        percentile_high = df_train[cols].sort_values(ascending=True).iloc[len(df_train[cols])//10*9]
    else:
        percentile_low = df_train[cols].describe()['25%']
        percentile_high = df_train[cols].describe()['75%']
    
    IQR = percentile_high - percentile_low
    lower_range = percentile_low - IQR * 1.5
    higher_range = percentile_high + IQR * 1.5
    df_train.loc[(df_train[cols] <= lower_range) | (df_train[cols] >= higher_range), cols] = np.nan



df_train = df_train.dropna()

important_col = filtered_df_tes_train.index.values
df_train2 = df_train.loc[:, df_train.columns.isin(important_col)].drop(columns=[targetColumn])
df_test2 = df_test.loc[:, df_test.columns.isin(important_col)]

df_train_target = df_train.loc[:, df_train.columns.isin([targetColumn])]
important_col = df_train2.columns





'''lasso = Ridge(alpha = 10)
lasso.fit(df_train2,df_train_target)'''

scaler = StandardScaler()
scaler.fit(df_train2)

df_train2 = scaler.transform(df_train2)
df_test2 = scaler.transform(df_test2)


params = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

ridge = Ridge()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(ridge, param_grid=params, cv=cv, scoring='neg_mean_absolute_error')
model.fit(df_train2, df_train_target)

# Predict and inverse-transform

y_pred = model.predict(df_test2)



df_test['Task5'] = y_pred
# Inițializăm o listă goală pentru a stoca rezultatele
result = []

# Iterăm prin fiecare rând al setului de date de test
for _, row in df_test.iterrows():
    # Iterăm prin subtasks (Task1 până la Task5)
    for subtask_id in range(1, 6):
        # Adăugăm un dicționar cu valorile corespunzătoare fiecărui subtask
        result.append({
            'subtaskID': subtask_id,  # ID-ul subtask-ului
            'datapointID': row['ID'],  # ID-ul datapoint-ului din rândul curent
            'answer': row[f'Task{subtask_id}']  # Răspunsul pentru subtask-ul curent
        })

# Creăm un DataFrame cu rezultatele obținute
df_output = pd.DataFrame(result)

# Afișăm primele 5 rânduri din DataFrame-ul rezultat
df_output.head()

# Salvăm rezultatele într-un fișier CSV pe care să-l putem apoi submite pe platformă
df_output.to_csv('submission.csv', index=False)

