# Importăm biblioteca pandas pentru a manipula datele
from scipy.stats import pointbiserialr
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier



label_encoder = preprocessing.LabelEncoder()

# Citim fișierul CSV cu datele de test și afișăm primele 5 rânduri
df_train = pd.read_csv('train_data.csv')
df_train.head()
target_column = 'Status'
df_train2 = df_train.copy()
df_train2[target_column]= label_encoder.fit_transform(df_train2[target_column])
# Citim fișierul CSV cu datele de antrenament și afișăm primele 5 rânduri
df_test = pd.read_csv('test_data.csv')
df_test.head()



# Adăugăm coloane noi în setul de date de test pentru fiecare subtask
df_test['Task1'] = "Unknown"
df_test['Task2'] = "Unknown"
df_test['Task3'] = 1
df_test['Task4'] = 0
df_test['Task5'] = "Alive"


df_test['Task1'] = df_test['GFR'].apply(lambda x: 'Normal' if x>=90 else 'Midly Decreasing' if x>=60  else 'Unknown')

value_25th_per = df_train['Serum Creatinine'].describe()['25%']
value_50th_per = df_train['Serum Creatinine'].describe()['50%']
value_75th_per = df_train['Serum Creatinine'].describe()['75%']

df_test['Task2'] = df_test['Serum Creatinine'].apply(lambda x: 'Very Low' if x <= value_25th_per else 'Low' if x<=value_50th_per else 'High' if x<=value_75th_per else 'Very High')


df_test['Task3'] = df_test['BMI'].apply(lambda x: 1 if x>value_50th_per else 0)
#df_test['Task4'] = df_test['T Stage'].apply(lambda x: df_test[df_test['T Stage']==x].value_counts())

values = df_test['T Stage'].unique()
for val in values:
    rows = df_test[df_test['T Stage'] == val].index
    df_test.loc[rows, 'Task4'] = len(df_test[df_test['T Stage'] == val]['T Stage'])                                                                   
# Inițializăm o listă goală pentru a stoca rezultatele
#de vizualizat elementele
important_columns = []
numerical_columns = df_train.select_dtypes(include='number').columns
df_test2 = df_test
for cols in df_train.select_dtypes(include='number').columns:
    if cols == target_column:
        continue
    z_scores = [(x -  df_train[cols].describe()['mean']) / df_train[cols].describe()['std'] for x in df_train['Serum Creatinine']]
    a = (df_train[cols]).describe()['std']

    if df_train[cols].describe()['std'] > 50:
        value_l = df_train[cols].dropna().sort_values().iloc[len(df_train[cols].dropna())//10]
        value_h = df_train[cols].dropna().sort_values().iloc[len(df_train[cols].dropna())//10*9]
    else:
        value_l = df_train[cols].describe()['25%']
        value_h = df_train[cols].describe()['75%']


    IQR = value_h - value_l
    lower_range = value_l - 1.5*IQR
    higher_range = value_h + 1.5*IQR
    number_elements = filter(lambda x: x, df_train2[cols].isna())

    corr, p = pointbiserialr(df_train2[cols], df_train2[target_column])
    if p < 0.05 and (p<0.00000001 or corr>0.1):
        important_columns.append(cols)



    df_train.loc[(df_train[cols] <= lower_range) | (df_train[cols] >= higher_range), cols] = np.nan
    df_train2.loc[(df_train2[cols] <= lower_range) | (df_train2[cols] >= higher_range), cols] = np.nan
all_columns = df_train.columns
cat_columns = all_columns.drop(numerical_columns)

for cols in cat_columns:
    df_train2[cols]= label_encoder.fit_transform(df_train2[cols])
    chisqt = pd.crosstab(df_train2[cols], df_train2[target_column], margins=True)
    value = np.array([chisqt.iloc[0].values,
                    chisqt.iloc[1].values])
    corr, p, freedom = chi2_contingency(value)[0:3]

    n = value.sum().sum()
    r,k = value.shape
    phi2 = corr/n
    phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1))
    rcorr = r - ((r - 1) ** 2) / ( n - 1 )
    kcorr = k - ((k - 1) ** 2) / ( n - 1 )
    try:
        if min((kcorr - 1),(rcorr - 1)) == 0:          
            v = 0
        else:
            v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    except:
        v = 0
    
    
    if p < 0.05:
        if v > 0.1:
            important_columns.append(cols)

df_train2 = df_train2.loc[:, important_columns]
df_train = df_train.loc[:, important_columns]
numerical_columns = df_train.select_dtypes(include='number').columns
df_train_target = df_train.loc[:, target_column]

df_numeric = df_train.loc[:, numerical_columns]

df_tes_train = df_numeric.corr(method='pearson')
for cols in df_tes_train.columns:
    if cols == target_column or cols not in list(df_train2.columns):
        continue

    filtered_df_tes_train2 = df_tes_train[((df_tes_train[cols]>=0.7) & (df_tes_train[cols]<1)) | ((df_tes_train[cols]<-0.7) & (df_tes_train[cols]<1))]
    if len(filtered_df_tes_train2) == 1 and filtered_df_tes_train2.index.values[0] == target_column:
        continue
    
    if len(filtered_df_tes_train2) > 0:
        first = abs(df_tes_train[target_column][cols])
        second = abs(df_tes_train[target_column][filtered_df_tes_train2.index.values[0]])

        if first > second:
            df_train2.drop(columns=[filtered_df_tes_train2.index.values[0]])
            df_train.drop(columns=[filtered_df_tes_train2.index.values[0]])
        else:
            df_train2.drop(columns=[cols])
            df_train.drop(columns=[cols])

numerical_columns = df_train.select_dtypes(include='number').columns
all_columns = df_train2.columns
cat_columns = all_columns.drop(numerical_columns).drop(target_column)
associated_columns = {}
for cols in cat_columns:
    for new_col in cat_columns:
        if cols == new_col:
            continue

        df_train2[cols]= label_encoder.fit_transform(df_train2[cols])
        df_test2[cols]= label_encoder.fit_transform(df_test[cols])

        chisqt = pd.crosstab(df_train2[cols], df_train2[new_col], margins=True)
        value = np.array([chisqt.iloc[0].values,
                        chisqt.iloc[1].values])
        try:
            corr, p, freedom = chi2_contingency(value)[0:3]
            n = value.sum().sum()
            r,k = value.shape
            phi2 = corr/n
            phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1))
            rcorr = r - ((r - 1) ** 2) / ( n - 1 )
            kcorr = k - ((k - 1) ** 2) / ( n - 1 )
            try:
                if min((kcorr - 1),(rcorr - 1)) == 0:          
                    v = 0
                else:
                    v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            except:
                v = 0
            
            if p < 0.05:
                if v > 0.3:
                                       
                    if cols not in associated_columns.keys():
                        associated_columns[cols] = [new_col]
                    else:
                        list_el = associated_columns[cols]
                        list_el.append(new_col)
                        associated_columns[cols] = list_el

                    if new_col not in associated_columns.keys():
                        associated_columns[new_col] = [cols]
                    else:
                        list_el = associated_columns[new_col]
                        list_el.append(cols)
                        associated_columns[new_col] = list_el

        except ValueError as Error:
            continue


for cols in cat_columns:
    if cols not in associated_columns.keys():
        continue
    associated_columns[cols] = set(associated_columns[cols])


things_to_drop = []
for cols in cat_columns:
    for new_col in cat_columns:
        if cols == new_col:
            continue

        df_train2[cols]= label_encoder.fit_transform(df_train2[cols])
        chisqt = pd.crosstab(df_train2[cols], df_train2[new_col], margins=True)
        value = np.array([chisqt.iloc[0].values,
                        chisqt.iloc[1].values])
        try:
            corr, p, freedom = chi2_contingency(value)[0:3]
            n = value.sum().sum()
            r,k = value.shape
            phi2 = corr/n
            phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1))
            rcorr = r - ((r - 1) ** 2) / ( n - 1 )
            kcorr = k - ((k - 1) ** 2) / ( n - 1 )
            try:
                if min((kcorr - 1),(rcorr - 1)) == 0:          
                    v = 0
                else:
                    v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            except:
                v = 0

            if p < 0.05:
                if v > 0.3:
                    size_a = len(associated_columns[cols])
                    size_b = len(associated_columns[new_col])
                    if size_a > size_b and cols not in things_to_drop:
                        things_to_drop.append(new_col)
                    
                    if size_b > size_a and new_col not in things_to_drop:
                        things_to_drop.append(cols)
                

        except ValueError as Error:
            continue


df_train2.dropna()

all_columns = cat_columns
all_columns = all_columns.append(numerical_columns)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_leaf_nodes=10)
important_columns.remove(target_column)
df_train2 = df_train2.loc[:, important_columns]
df_train = df_train.loc[:, important_columns]
df_test2 = df_test2.loc[:, important_columns]

X_train = df_train2
y_train = df_train_target
clf = clf.fit(X_train,y_train)
X_test = df_test2

y_pred = clf.predict(X_test)

df_test['Task5'] = y_pred 
result = []

# Iterăm prin fiecare rând al setului de date de test
for _, row in df_test.iterrows():
    # Iterăm prin subtasks (Task1 până la Task5)
    for subtask_id in range(1, 6):
        # Adăugăm un dicționar u valorile corespunzătoare fiecărui subtask
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
