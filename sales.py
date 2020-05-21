import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("./Sales_Data/Sales_April_2019.csv")
files = [file for file in os.listdir('./Sales_Data')]
all_months_data = pd.DataFrame()
for file in files:
    df = pd.read_csv("./Sales_Data/" + file)
    all_months_data = pd.concat([all_months_data, df])

all_months_data.to_csv("all_data.csv", index=False)
all_data = pd.read_csv("all_data.csv")
nan_df = all_data[all_data.isna().any(axis=1)]
all_data = all_data.dropna(how='all')
# print(all_data)
# Drop Duplicated column headers
all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']
all_data["Month"] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')

all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])
all_data['Sale'] = all_data['Quantity Ordered'] * all_data['Price Each']


# add a city column (.apply ()) method
def get_city(address):
    return address.split(',')[1]


def get_state(address):
    return address.split(',')[2].split(' ')[1]


# all_data['Column']=all_data['Purchase Address'].apply(lambda x: x.split(',')[1])
all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")
all_data.head()
#print(all_data.head())

# results = all_data.groupby('Month').sum()

# months = range(1, 13)
# print(plt.bar(months, results['Sale']))
# plt.xticks(months)
# plt.ylabel('Sales in USD ($)')
# plt.xlabel('Month Number')
# all_data.head()
#results = all_data.groupby('City').sum()
#print(results)

#cities = [city for city, df in all_data.groupby('City')]
#print(plt.bar(cities, results['Sale']))
#plt.xticks(cities, rotation='vertical', size=8)
#plt.ylabel('Sales in USD ($)')
#plt.xlabel('Name of City')
#plt.show()
# all_data.head()

all_data['Order Date']=pd.to_datetime(all_data['Order Date'])
all_data['Hour']=all_data['Order Date'].dt.hour
all_data['Minute']=all_data['Order Date'].dt.minute


#hours = [hour for hour, df in all_data.groupby('Hour')]
#plt.plot(hours, all_data.groupby(['Hour']).count())
#print(all_data.groupby(['Hour']).count())
#plt.xticks(hours)
#plt.ylabel('Hour')
#plt.xlabel('Number of Orders')
#plt.grid()
#plt.show()
#print(all_data.head())

df = all_data[all_data['Order ID'].duplicated(keep=False)]
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
df=df[['Order ID', 'Grouped']].drop_duplicates()
#print(df.head())


from itertools import combinations
from collections import Counter
count = Counter()
for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

for key, value in count.most_common(10):
    print(key, value)


product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum () ['Quantity Ordered']
products = [product for product, df in product_group]
plt.bar(products, quantity_ordered)
plt.xticks(products, rotation='vertical', size=8)
plt.ylabel('Quantity Ordered')
plt.xlabel('Product')
#plt.show()

prices = all_data.groupby('Product').mean()['Price Each']
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='green')
ax2.plot(products, prices, 'b-')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price ($)', color='b')
ax1.set_xticklabels(products, rotation='vertical', size=8)
plt.show()
print(prices)