import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


df = pd.read_csv("AmesHousing.csv")

pd.set_option('display.max_rows', None)

print(df.head())
print(df.shape)
print(df.info())

#Очистить данные от пропусков (В колонках типа Alley, Pool QC, Fence – замените NaN на строку "None"
#отсутствие объекта). В числовых колонках Bsmt Full Bath, Garage Area – NaN означает отсутствие подвала/гаража → замените на 0.
#В Lot Frontage – заполните медианой по соседству (Neighborhood) и т.д.)

def print_missing_info(df):
    missing_info = pd.DataFrame({
        'NaN_count': df.isna().sum(),
        'dtype': df.dtypes
    })

    missing_info = missing_info.sort_values(
        by='NaN_count',
        ascending=False
    )

    print(missing_info)

print_missing_info(df)

zero_cols = [
  'Mas Vnr Area',
  'BsmtFin SF 1',
  'BsmtFin SF 2',
  'Bsmt Unf SF',
  'Total Bsmt SF',
  'Bsmt Full Bath',
  'Bsmt Half Bath',
  'Garage Cars',
  'Garage Area',  
]

df[zero_cols] = df[zero_cols].fillna(0)
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))
df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(df['Year Built'])

none_cols = [
  'Pool QC',
  'Misc Feature',
  'Alley',
  'Fence',
  'Mas Vnr Type',
  'Fireplace Qu',
  'Garage Cond',
  'Garage Finish',
  'Garage Qual',
  'Garage Type',
  'Bsmt Exposure',
  'BsmtFin Type 2',
  'Bsmt Qual',
  'Bsmt Cond', 
  'BsmtFin Type 1',
  'Electrical'   
]

df[none_cols] = df[none_cols].fillna("None")

print_missing_info(df)


# обработать категориальные признаки (One-Hot Encoding)

df = pd.get_dummies(df, drop_first=True)
print(df.shape)


# Для линейной модели (Ridge) постройте топ-10 самых важных признаков по модулю коэффициента

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

coefficients = pd.Series(ridge.coef_, index=X.columns)
top_10 = coefficients.abs().sort_values(ascending=False).head(10)


top_10.sort_values().plot(kind='barh')
plt.title('Top-10 признаков')
plt.ylabel('Признаки')
plt.xlabel('Абсолютное значение коэффициента')
plt.show()


#Визуализировать зависимость цены от жилой площади (Gr Liv Area). Использовать статистические методы
#(Z-score, IQR) или алгоритм Isolation Forest, чтобы выявить дома, которые стоят подозрительно дешево при огромной площади.

plt.scatter(df['Gr Liv Area'],df['SalePrice'])
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.title('Зависимость цены от жилой площади')
plt.show()

Q1 = df['Gr Liv Area'].quantile(0.25)
Q3 = df['Gr Liv Area'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[~((df['Gr Liv Area'] > upper_bound) & (df['SalePrice'] < 300000))]


#Сравнить качество регрессии «до» и «после» удаления аномалий.

y_pred_normal = ridge.predict(X_test_scaled)
rmse_normal = np.sqrt(mean_squared_error(y_test, y_pred_normal))
r2_normal = r2_score(y_test, y_pred_normal)

X_clean = df_no_outliers.drop('SalePrice', axis=1)
y_clean = df_no_outliers['SalePrice']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

ridge_clean = Ridge(alpha=1.0)
ridge_clean.fit(X_train_c_scaled, y_train_c)

y_pred_clean = ridge_clean.predict(X_test_c_scaled)
rmse_clean = np.sqrt(mean_squared_error(y_test_c, y_pred_clean))
r2_clean = r2_score(y_test_c, y_pred_clean)

print()
print("До удаления аномалий:")
print("RMSE:", rmse_normal)
print("R²:", r2_normal)
print()
print("После удаления аномалий:")
print("RMSE:", rmse_clean)
print("R²:", r2_clean)

plt.scatter(df_no_outliers['Gr Liv Area'],df_no_outliers['SalePrice'])
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.title('Зависимость цены от жилой площади')
plt.show()


# Разделить все объекты недвижимости на 4–5 логических групп (сегментов) без учета цены.

X_cluster = df_no_outliers.drop('SalePrice', axis=1)
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

df_no_outliers['Cluster'] = clusters

print(df_no_outliers['Cluster'].value_counts())

cluster_summary = df_no_outliers.groupby('Cluster')[
    [
        'Gr Liv Area',
        'Overall Qual',
        'Year Built',
        'Garage Area',
        'Total Bsmt SF',
        'Lot Area'
    ]
].mean()

print()
print(cluster_summary)


# Применить PCA (Метод главных компонент) ко всем числовым признакам. На полученных компонентах обучить любую модель регрессии.

X_pca = df_no_outliers.drop(['SalePrice', 'Cluster'],axis=1)

y_pca = df_no_outliers['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X_pca,y_pca,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print()
print(X_train_scaled.shape)
print(X_train_pca.shape)

ridge_pca = Ridge(alpha=1.0)
ridge_pca.fit(X_train_pca, y_train)

y_pred_pca = ridge_pca.predict(X_test_pca)

rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))
r2_pca = r2_score(y_test, y_pred_pca)

print()
print("До PCA:")
print("RMSE:", rmse_clean)
print("R²:", r2_clean)
print()
print("После PCA:")
print("RMSE:", rmse_pca)
print("R²:", r2_pca)


#Данные охватывают период с 2006 по 2010 год. Нужно проанализировать динамику цен по месяцам
#(Mo Sold) и годам (Yr Sold). Создать признак «возраст дома на момент продажи» и «лет с последнего ремонта».
#Понять, упали ли цены во время кризиса 2008 года и насколько сильно сезонность (продажа весной vs зимой) влияет на итоговый чек.

month_prices = df_no_outliers.groupby('Mo Sold')['SalePrice'].mean()
month_prices.plot(marker='o')
plt.title('Средняя цена домов по месяцам')
plt.xlabel('Месяц продажи')
plt.ylabel('Средняя цена')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

year_prices = df_no_outliers.groupby('Yr Sold')['SalePrice'].mean()
year_prices.plot(marker='o')
plt.title('Средняя цена домов по годам')
plt.xlabel('Год продажи')
plt.ylabel('Средняя цена')
plt.xticks(year_prices.index)
plt.grid(True)
plt.show()

df_no_outliers['House Age'] = (df_no_outliers['Yr Sold'] - df_no_outliers['Year Built'])
df_no_outliers['Years Since Remodel'] = (df_no_outliers['Yr Sold'] - df_no_outliers['Year Remod/Add'])
print(df_no_outliers[['House Age','Years Since Remodel']].head())