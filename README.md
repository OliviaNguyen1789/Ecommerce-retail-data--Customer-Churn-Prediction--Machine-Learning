# [Machine Learning]: E-commerce Customer Churn Prediction

Author: [Uyen Nguyen]  

Date: March 2025

Tools Used: Machine learning

---

## 📑 Table of Contents  
I. [Introduction](#i-introduction)  
II. [Dataset Description](#ii-dataset-description)  
III.[Exploring churn users' behaviour](#iii-exploring-churn-users'-behaviour)  
IV.[ML model for predicting churned users ](#iv-ml-model-for-predicting-churned-users)  
V.[ML model for segmenting churned users](#v-ml-model-for-segmenting-churned-users)  


## I. Introduction

### Project Aim:
- Develop a Random Forest machine learning model to predict churned users for an e-commerce company.
- Use K-means clustering to segment churned users and recommend tailored promotions based on their behavior.
  
### Stakeholders: 
- Data analysts & business analysts
- Marketing team.


## II. Dataset Description

- Source: The attached dataset provides customer information for an e-commerce company
- Size: 5630 rows, 20columns
- Format: .xlsx
- Table schema:

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






## III. Exploring churn users' behaviour

### 1. Data Cleaning

#### Handle missing values

<img width="599" alt="Screen Shot 2025-03-07 at 11 33 43 PM" src="https://github.com/user-attachments/assets/0a8ecd47-73cb-4ed6-afb0-b3af71b7e3af" />

Because % mising values > 20% and disperse in many columns --> using ML model to impute missing value

<img width="662" alt="Screen Shot 2025-03-07 at 8 58 08 PM" src="https://github.com/user-attachments/assets/4dcc6803-e05c-45b6-b13d-4e7cbe35b11f" />


#### Handle duplicate values

<img width="489" alt="Screen Shot 2025-03-07 at 8 53 21 PM" src="https://github.com/user-attachments/assets/35b07fc1-e5d7-4dba-b4dc-f3b5d3813f5b" />

The dataset has no duplicate values.

#### Univariate Analysis

<img width="641" alt="Screen Shot 2025-03-07 at 8 54 18 PM" src="https://github.com/user-attachments/assets/42879528-33ef-4074-a7e0-856ccd8abc83" />

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

<img width="578" alt="Screen Shot 2025-03-07 at 8 55 20 PM" src="https://github.com/user-attachments/assets/19527cb3-60c9-4350-b85b-05f6ac046091" />

### 2. Feature engineering

<img width="901" alt="Screen Shot 2025-03-07 at 9 05 07 PM" src="https://github.com/user-attachments/assets/d2385cee-487b-452c-8290-5c9137ac1fe2" />

### 3. Training Random Forest model

<img width="816" alt="Screen Shot 2025-03-07 at 11 38 55 PM" src="https://github.com/user-attachments/assets/5ef9d20d-97c9-4c1e-9334-3db841a682d8" />

### 4. Find out feature importance

<img width="821" alt="Screen Shot 2025-03-07 at 11 46 03 PM" src="https://github.com/user-attachments/assets/d7c4ba95-f6c3-4cba-b6b9-d98ab9708de6" />

<img width="975" alt="Screen Shot 2025-03-07 at 11 46 26 PM" src="https://github.com/user-attachments/assets/f0f29c5d-6ce9-4cd2-8730-944183d5a92a" />

The figure shows that Tenure, CashbackAmount, WarehousetoHome, Complain, DaysinceLastOrder have a strong impact on churn.

#### Explain the Impact of Feature importance

<img width="927" alt="Screen Shot 2025-03-07 at 11 50 17 PM" src="https://github.com/user-attachments/assets/12f2343a-df7e-4a93-ac0a-39a037021eb1" />

  
| Insights                                                        | Recommendations                                                                                                                                           |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| New customers --> higher churn risk                  | Boost new customer engagement through personalized recommendations and exclusive discounts, fostering long-term brand loyalty.|
| Lower cash back amount --> higher churn risk         | Offer increasing cashback percentages based on total spending, encouraging repeat purchases and long-term engagement.         |
| Higher distance from warehouse --> higher churn risk | Offer free/discounted delivery for high-risk customers; Optimize fulfillment centers.                                         |
| High Complain --> higher churn risk                  | Build long-term customer trust by addressing complaints swiftly, following up proactively, and offering personalized solutions that enhance satisfaction. |
| Customers with recent orders --> higher churn risk   | Implement a quick feedback mechanism after purchase/delivery and analyze this trend across customer segments.                 |

## IV. ML model for predicting churned users 

### 1. Training different models

<img width="771" alt="Screen Shot 2025-03-08 at 12 14 46 AM" src="https://github.com/user-attachments/assets/bafc8619-b533-410c-a23c-a90685341008" />

### 2. Model evaluation

<img width="497" alt="Screen Shot 2025-03-08 at 12 15 15 AM" src="https://github.com/user-attachments/assets/ad0d64e8-c56b-422e-ace4-0c2d114d4198" />
 
Apparently, Random Forest model offers the highest f1_score, so it is considered as base model.

### 3. Improve model 
To enhance churn prediction model, we will perform hyperparameter tuning with GridSearchCV to find the optimal parameter combination for better performance.

<img width="1104" alt="Screen Shot 2025-03-08 at 12 46 00 AM" src="https://github.com/user-attachments/assets/18bb2fa9-5ae0-46ea-a892-98dff28d657b" />


## V. ML model for segmenting churned users

### 1. Create Churn user dataset
<img width="623" alt="Screen Shot 2025-03-08 at 12 22 03 AM" src="https://github.com/user-attachments/assets/c9105809-8eba-4d4e-9781-044553e313c0" />

### 2. Select the number of cluster
To determine the optimal number of clusters for segmenting our churned customers, I employ the Elbow Method using K-means clustering.

<img width="597" alt="Screen Shot 2025-03-08 at 12 23 35 AM" src="https://github.com/user-attachments/assets/ebcd1bc8-619c-4226-b08c-6d6f189d86d3" />

<img width="644" alt="Screen Shot 2025-03-08 at 12 23 56 AM" src="https://github.com/user-attachments/assets/1e203bc5-aaed-46d4-87f0-11654f9580c4" />

The inertia decreases slowly from 5 --> the number of cluster = 5

### 3. Model training

<img width="558" alt="Screen Shot 2025-03-08 at 12 25 00 AM" src="https://github.com/user-attachments/assets/d979e3b1-2e88-4753-b883-4f566af1930c" />

We calculate the mean values of five key features for each cluster to identify distinct patterns and behaviors within groups of churned customers

<img width="797" alt="Screen Shot 2025-03-08 at 12 48 08 AM" src="https://github.com/user-attachments/assets/5548c3de-7893-4876-83af-49351734ca55" />


### 4.Segmentation and Recommendations for Promotion

#### The characteristic of each segments

Based on the clustering data provided for churned users, we can identify 5 distinct segments.

- Segment 1: High-Value, Satisfied but Inactive users
- Segment 2: High-Value, quite-Satisfied and Active users.
- Segment 3 : High-Value but Frequent complaints users.
- Segment 4: New commers): New, Satisfied but Low-Value users.
- Segment 5 (Occasional Buyers): Low-Value, engagement but frequent complaints.


#### Promotion Strategy:

- Segment 1: Provide personalized recommendations & exclusive Loyalty Discount; Provide a cashback incentive for returning customers.
- Segment 2: Free or Discounted Express Shipping, Personalized Product Recommendations & Bundles.
- Segment 3: free or reduce delivery cost, VIP Customer Care & Priority Support, Surprise "Thank You" Gift or Loyalty Perks
- Segment 4: Time-Limited Bonus Coupon for new customers, Free Gift with Purchase Over Minimum Amount.
- Segment 5: Provide exclusive trial offers and Limited-Time Flash Sale to boost the engagement, Offer customer support follow-up & service improvements.

