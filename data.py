from typing import Any, Union

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.linear_model import LinearRegression

df: Union[Union[TextFileReader, Series, DataFrame, None], Any] = pd.read_csv('dataset.csv')

outliers = []

df.describe()
print(df.describe())

def detect_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers


# outlier_rr = detect_outliers(df['rr_60'])
# print(outlier_rr)
# outlier_temp = detect_outliers(df['temp_f'])
# print(outlier_temp)
# outlier_o2satf = detect_outliers(df['o2_satf'])
# print(outlier_o2satf)

# sorted(df['rr_60'])
# print(sorted(df['rr_60']))

# print(df['o2_satf'])


# Outliers for Oxygen Saturation

q1, q3 = np.percentile(df['o2_satf'], [25, 75])
iqr = q3 - q1
# print(iqr)
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

# print(lower_bound)
# print(upper_bound)

row_indexes = df[df['o2_satf'] >= 105.5].index
df.loc[row_indexes, 'outlier_o2'] = "yes"

row_indexes = df[df['o2_satf'] <= 77.5].index
df.loc[row_indexes, 'outlier_o2'] = "yes"

# OUTLIERS FOR RESPIRATORY RATES
q1, q3 = np.percentile(df['rr_60'], [25, 75])
iqr = q3 - q1
# print(iqr)
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

# print(lower_bound)
# print(upper_bound)

row_indexes = df[df['rr_60'] >= 40].index
df.loc[row_indexes, 'outlier_rr'] = "yes"

row_indexes = df[df['rr_60'] <= 8].index
df.loc[row_indexes, 'outlier_rr'] = "yes"

# OUTLIERS FOR Temperature
q1, q3 = np.percentile(df['temp_f'], [25, 75])
iqr = q3 - q1
# print(iqr)
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

# print(lower_bound)
# print(upper_bound)

row_indexes = df[df['temp_f'] >= 105.5].index
df.loc[row_indexes, 'outlier_t'] = "yes"

row_indexes = df[df['temp_f'] <= 93.5].index
df.loc[row_indexes, 'outlier_t'] = "yes"

# Linear Regression_Temperature

x = df['temp_f'].values.reshape(-1, 1)
y = df['risk_pts'].values.reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, y)
y_pred = lr.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.ylabel('Pulmonary Severity Index')
plt.xlabel('Temperature')
# plt.show()

# Linear Regression_Respiratory Rate
x = df['rr_60'].values.reshape(-1, 1)
y = df['risk_pts'].values.reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, y)
y_pred = lr.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.ylabel('Pulmonary Severity Index')
plt.xlabel('Respiratory Rate')
# plt.show()


# Linear Regression_Oxygen Saturation
x = df['o2_satf'].values.reshape(-1, 1)
y = df['risk_pts'].values.reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, y)
y_pred = lr.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')


plt.ylabel('Pulmonary Severity Index')
plt.xlabel('Oxygen Saturation')
# plt.show()


# create linear regression object
mlr = LinearRegression()

# fit linear regression
mlr.fit(df[['rr_60', 'temp_f', 'o2_satf']], df['risk_pts'])

# get the slope and intercept of the line best fit.
print(mlr.intercept_)
print(mlr.coef_)

# regression plot using seaborn
fig = plt.figure(figsize=(10, 7))
# sns.residplot(x=df['temp_f'], y=df['risk_pts'], color='blue', marker='+')
# sns.residplot(x=df['rr_60'], y=df['risk_pts'], color='magenta', marker='+')
# sns.residplot(x=df['o2_satf'], y=df['risk_pts'], color='red', marker='+')

# sns.residplot(x='rr_60', y='risk_pts', data=df, color='brown')
sns.residplot(x='o2_satf', y='risk_pts', data=df, color='black')
# sns.residplot(x='temp_f', y='risk_pts', data=df, color='red'),
scatter_kws = {'s': 10};

plt.show()
# Legend, title and labels.
# plt.legend(labels=['Males', 'Females'])
# plt.title('Relationship between Height and Weight', size=24)
# plt.xlabel('Height (inches)', size=18)
# plt.ylabel('Weight (pounds)', size=18);
# plt.sho