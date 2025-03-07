# -Machine-Learning-Churn-Prediction-

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
VI. [Final Conclusion & Recommendations](#vi-final-conclusion--recommendations)

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











## IV. ML model for predicting churned users 


## V. ML model for segmenting churned users

## VI. Final Conclusion & Recommendations 

The analysis has revealed some inefficiencies in manufacturing performance. The most critical problems are:



These issues directly impact manufacturing efficiency, delivery performance, and cost control, making them a priority for process improvement.

**Recommendations:**

