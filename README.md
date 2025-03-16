
<img width="1000" alt="Screen Shot 2025-03-16 at 3 00 20 PM" src="https://github.com/user-attachments/assets/4cdfa840-e6e3-40be-ab32-f4acdaf6726b" />


# Project Title: Ecommerce retail data - Customer Churn Prediction | Machine Learning

Author: [Uyen Nguyen]  
Date: March 2025  
Tools Used: Machine learning

---

## 📑 Table of Contents  
I. [📌 Background & Overview](#-background--overview)  
II. [📂 Dataset Description](#-dataset-description)  
III. [📊 Exploring churn users' behaviour](#-exploring-churn-users-behaviour)  
IV. [ ⚒️ ML model for predicting churned users](#-ml-model-for-predicting-churned-users)  
V. [⚒️ ML model for segmenting churned users](#-#%EF%B8%8F-ml-model-for-segmenting-churned-users)

## 📌 Background & Overview

### 📖 What is this project about?
The objective of this project is to analyze a Bicycle Manufacturer dataset and develop a machine learning model to predict customer churn. This involves leveraging machine learning techniques, including Random Forest and K-means clustering. The insights gained will help stakeholders design effective retention strategies and minimize customer attrition.
  
### 👤 Who is this project for?
- Data analysts & business analysts
- Marketing team.


## 📂 Dataset Description

### 🌐 Data Source  
- The attached dataset provides customer information for an e-commerce company
- Size: 5630 rows, 20columns
- Format: .xlsx

### 🔀 Table schema
<details>
<summary>Table: Churn Prediction </summary>  

| Variable                     | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| CustomerID                   | Unique customer ID                                           |
| Churn                        | Churn Flag                                                   |
| Tenure                       | Tenure of customer in organization                           |
| PreferredLoginDevice         | Preferred login device of customer                           |
| CityTier                     | City tier (1,2,3)                                            |
| WarehouseToHome              | Distance in between warehouse to home of customer            |
| PreferPaymentMethod          | PreferredPaymentMode Preferred payment method of customer    |
| Gender                       | Gender of customer                                           |
| HourSpendOnApp               | Number of hours spend on mobile application or website       |
| NumberOfDeviceRegist ered    | Total number of devices is registered on particular customer |
| PreferedOrderCat             | Preferred order category of customer in last month           |
| SatisfactionScore            | Satisfactory score of customer on service                    |
| MaritalStatus                | Marital status of customer                                   |
| NumberOfAddress              | Total number of added added on particular customer           |
| Complain                     | Any complaint has been raised in last month                  |
| OrderAmountHikeFroml astYear | Percentage increases in order from last year                 |
| CouponUsed                   | Total number of coupon has been used in last month           |
| OrderCount                   | Total number of orders has been places in last month         |
| DaySinceLastOrder            | Day Since last order by customer                             |
| CashbackAmount               | Average cashback in last month                               |

</details>

## 📊 Exploring churn users' behaviour

### 1️⃣ Data Cleaning

#### Handle missing values

```sql
# Check % missing values
missing_row_percentage = df_raw.isnull().any(axis=1).mean()*100
print(missing_row_percentage)
```
32.96625222024866

```sql
# Check columns having missing values
missing_col=df_raw.isnull().sum()
missing_col = missing_col[missing_col>0]
missing_col
```
Because % mising values > 20% and disperse in many columns --> using ML model to impute missing value

```sql
# Data type of missing columns is float --> imputer mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_raw[missing_col.index] = imputer.fit_transform(df_raw[missing_col.index])
df_raw.isnull().sum()
```

#### Handle duplicate values

```sql
# Check duplicate
duplicate_count = df_raw.duplicated('CustomerID').sum()
print(duplicate_count)
```

The dataset has no duplicate values.

#### Univariate Analysis

```sql
# Categorical data:
cate_cols = df_raw.loc[:, df_raw.dtypes == object].columns.tolist()
for col in cate_cols:
    print(f"Unique values of {col}: {df_raw[col].nunique()}")
```

```sql
# Numerical data:
numeric_cols = df_raw.loc[:, df_raw.dtypes != object].columns.tolist()
for col in numeric_cols:
    print(f"Unique values of {col}: {df_raw[col].nunique()}")

for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_raw[col], color='#86D293')
    plt.title(f'Boxplot of {col}')
    plt.show()
```
<img width="927" alt="Screen Shot 2025-03-16 at 3 40 01 PM" src="https://github.com/user-attachments/assets/9591d63f-71af-4388-9ac6-905622c3933a" />


*   Tenure: right-skewed, most customers clustered in 0 - 20
*   WarehouseToHome: majority of customers having short distances (below 40).
*   HourSpendOnApp: centered around 2-3 hours
*   NumberOfDeviceRegistered: centered around 3-4 devices
*   SatisfactionScore: mean at 3, suggesting generally moderate to positive customer satisfaction levels.
*   NumberOfAddress : most customers have 1-5 addresses, several outliers having up to 22 addresses 
*   OrderAmountHikeFromlastYear: growth in customer spending, centered around 13-18%
*   CouponUsed: most customers using 0-2 coupons, some outliers using up to 16
*   OrderCount: most customers placing 1-3 orders and several outliers up to 16 orders
*   DaySinceLastOrder: Most customers have ordered recently (within 0-10 days)
*   CashbackAmount: cashback distribution is complex, a cluster of high outliers around 300


#### Outlier Detection
Remove the columns that have a few outliers causing the data to be skewed:
* Tenuere: >40
* WarehouseToHome >100
* NumberOfAddress >15
* DaySinceLastOrder >20

```sql
df = df_raw[
    (df_raw['Tenure'] <= 40) &
    (df_raw['WarehouseToHome'] <= 100) &
    (df_raw['NumberOfAddress'] <= 15) &
    (df_raw['DaySinceLastOrder'] <= 20)
]
df = df.reset_index(drop=True)
# Check rows_removed
len(df_raw) - len(df)
```

### 2️⃣ Feature engineering

```sql
# Encoding columns:
col_encoding=['PreferredLoginDevice','PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus']
df_encoding = pd.get_dummies(df, columns= col_encoding,drop_first=True)
df_encoding.head()
```

### 3️⃣ Training Random Forest model

```sql
from sklearn.model_selection import train_test_split

X = df_encoding.drop('Churn', axis=1)
y = df_encoding['Churn']

# Split data
X_train, X_tem, y_train, y_tem = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tem, y_tem, test_size=0.5, random_state=42)
print(f"Number data of train set: {len(X_train)}")
print(f"Number data of validate set: {len(X_val)}")
print(f"Number data of test set: {len(X_test)}")

# Normalization

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier
rd = RandomForestClassifier(random_state=42, n_estimators = 100)
rd.fit(X_train_scaled, y_train)

y_rd_pred = rd.predict(X_val_scaled) #model to predict
y_rd_pred_train = rd.predict(X_train_scaled) #Predict back on train to check overfit
y_rd_pred_val = rd.predict(X_val_scaled)  ##Predict on validate dataset
```

### 4️⃣ Find out feature importance
```sql
# Get feature importance
importances = rd.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#6256CA')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance in Churn Prediction')
plt.gca().invert_yaxis()
plt.show()
```

<img width="975" alt="Screen Shot 2025-03-07 at 11 46 26 PM" src="https://github.com/user-attachments/assets/f0f29c5d-6ce9-4cd2-8730-944183d5a92a" />

The figure shows that Tenure, CashbackAmount, WarehousetoHome, Complain, DaysinceLastOrder have a strong impact on churn.

#### Explain the Impact of Feature importance

```sql
feature_draw = df_encoding[['Tenure', 'CashbackAmount', 'WarehouseToHome', 'Complain', 'DaySinceLastOrder']]

for col in feature_draw.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Churn', y=col, data=df_encoding, palette={'0': '#6256CA', '1': '#86D293'})
    plt.title(f'Boxplot of {col}')
    plt.show()
```

<img width="974" alt="Screen Shot 2025-03-16 at 3 48 19 PM" src="https://github.com/user-attachments/assets/9b1ef15c-e1d2-4e79-bec7-6b8534e5c03c" />

  
| Insights                                            | Recommendations                                                                                                       |
| ----------------------------------------------------| --------------------------------------------------------------------------------------------------------------------- |
| New customers --> higher churn risk                 | Boost new customer engagement through personalized recommendations and exclusive discounts, fostering long-term brand loyalty.|
| Lower cash back amount --> higher churn risk        | Offer increasing cashback percentages based on total spending, encouraging repeat purchases and long-term engagement. |
| Higher distance from warehouse --> higher churn risk| Offer free/discounted delivery for high-risk customers; Optimize fulfillment centers.                                 |
| High Complain --> higher churn risk                 | Build long-term customer trust by addressing complaints swiftly, following up proactively, and offering personalized olutions that enhance satisfaction. |
| Customers with recent orders --> higher churn risk  | Implement a quick feedback mechanism after purchase/delivery and analyze this trend across customer segments.         |

## ⚒️ ML model for predicting churned users

### 1️⃣ Training different models

```sql
#knn model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train) #model to learn
y_knn_pred_val = knn.predict(X_val_scaled) #Predict on validate dataset
y_knn_pred_train = knn.predict(X_train_scaled) #Predict back on train to check overfit
```

```sql
# Logistic regression model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train) #model to learn
y_lr_pred_val = lr.predict(X_val_scaled) #Predict on validate dataset
y_lr_pred_train = lr.predict(X_train_scaled) #Predict back on train to check overfit
```

```sql
# Random Forest model
from sklearn.ensemble import RandomForestClassifier

rd = RandomForestClassifier(random_state=42, n_estimators = 100)
rd.fit(X_train_scaled, y_train)

y_rd_pred_val = rd.predict(X_val_scaled)  ##Predict on validate dataset
y_rd_pred_train = rd.predict(X_train_scaled) #Predict back on train to check overfit
```

```sql
#AdaBoost (Adaptive Boosting)
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada.fit(X_train_scaled, y_train)
y_ada_pred_val = ada.predict(X_val_scaled)
y_ada_pred_train = ada.predict(X_train_scaled)
```
### 2️⃣ Model evaluation

```sql
#knn model
from sklearn.metrics import f1_score
f1_score_train = f1_score(y_train, y_knn_pred_train)
f1_score_val = f1_score(y_val, y_knn_pred_val)
print(f1_score_train, f1_score_val)
```
0.7458745874587459 0.6095238095238096

```sql
#Logistic regression model
from sklearn.metrics import f1_score
f1_score_train = f1_score(y_train, y_lr_pred_train)
f1_score_val = f1_score(y_val, y_lr_pred_val)
print(f1_score_train, f1_score_val)
```
0.5985267034990792 0.647887323943662

```sql
#random forest
from sklearn.metrics import f1_score
f1_score_train = f1_score (y_train, y_rd_pred_train)
f1_score_val = f1_score(y_val, y_rd_pred_val)
print(f1_score_train, f1_score_val)
```
1.0 0.8888888888888888

```sql
#AdaBoost (Adaptive Boosting)
from sklearn.metrics import f1_score
f1_score_train = f1_score (y_train, y_ada_pred_train)
f1_score_val = f1_score(y_val, y_ada_pred_val)
print(f1_score_train, f1_score_val)
```
0.62751677852349 0.6636771300448431

 
Apparently, Random Forest model offers the highest f1_score, so it is considered as base model.

### 3️⃣ Improve model 
To enhance churn prediction model, we will perform hyperparameter tuning with GridSearchCV to find the optimal parameter combination for better performance.

```sql
# Use GridSearchCV to find the best parameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 100, 200, 500],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid= param_grid, cv=5, scoring='f1')

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(X_test, y_test)
```
Best Parameters:  {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

## ⚒️ ML model for segmenting churned users

### 1️⃣ Create Churn user dataset

```sql
# Create churn users datatset
df_churn = df_encoding[df_encoding['Churn']==1]
df_churn.head()
```

```sql
# Normalizaiton
X1 = df_churn.drop('Churn', axis=1)
y1 = df_churn['Churn']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X1_scaled = scaler.fit_transform(X1)
X1_scaled = pd.DataFrame(X1_scaled, columns=X1.columns)
X1_scaled.head()
```
### 2️⃣ Select the number of cluster
To determine the optimal number of clusters for segmenting our churned customers, I employ the Elbow Method using K-means clustering.

```sql
# Find the number of cluster

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
k = range(1, 10)
inertia = []
for i in k:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X1_scaled)
    inertia.append(km.inertia_)
plt.plot(k, inertia, '-o')
plt.xlabel('number of cluster')
plt.ylabel('inertia')
plt.title('Elbow method showing the optimal k')
plt.show()
```

<img width="644" alt="Screen Shot 2025-03-08 at 12 23 56 AM" src="https://github.com/user-attachments/assets/1e203bc5-aaed-46d4-87f0-11654f9580c4" />

The inertia decreases slowly from 5 --> the number of cluster = 5

### 3️⃣ Model training
```sql
 Train K-Means model with optimal K

from sklearn.cluster import KMeans
km = KMeans(n_clusters=5, random_state=42)

# Assign Customers into different segments

df_churn['Customer_segment'] = km.fit_predict(X1_scaled)
df_churn.head()
```

We calculate the mean values of five key features for each cluster to identify distinct patterns and behaviors within groups of churned customers
```sql
features = ['Tenure', 'CashbackAmount', 'WarehouseToHome', 'Complain', 'DaySinceLastOrder']
# Get average feature values per segment
segment_analysis = df_churn.groupby('Customer_segment')[features].mean()
print(segment_analysis)
```
<img width="588" alt="Screen Shot 2025-03-09 at 12 58 26 AM" src="https://github.com/user-attachments/assets/f51b5512-8249-486d-b891-fee12b3b8dcf" />

### 4️⃣ egmentation and Recommendations for Promotion

Based on the clustering data for churned users, we have identified five distinct segments and proposed tailored promotion strategies for each:

| Segment | Characteristics                                                | Promotion Strategy                                                                                               |
| ------- | ---------------------------------------------------------------| ---------------------------------------------------------------------------------------------------------------- |
|    1    | High-Value, Satisfied but Inactive Users                       | Personalized recommendations & exclusive loyalty discounts, cashback incentives for returning customers.         |
|    2    | High-Value, Quite-Satisfied and Active Users                   | Free or discounted express shipping, personalized product recommendations & bundles.                             |
|    3    | High-Value but Frequent complaints users                       | Free or reduced delivery costs, VIP customer care & priority support, surprise "Thank You" gift or loyalty perks |
|    4    | Newcomers – New, Satisfied but Low-Value Users                 | Time-limited bonus coupon for new customers, free gift with purchase over a minimum amount.                      |
|    5    | Occasional Buyers – Low-Value, Engaged but Frequent Complaints | Exclusive trial offers and limited-time flash sales to boost engagement, customer support follow-up & service improvements. |

