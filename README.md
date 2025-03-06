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

Based on the AdventureWorks database, this project analyzes the manufacturing process of a bicycle manufacturer using Power BI.
### The objective:
- Ensure products are delivered on time.
- Minimize scrapped products.
  
### Stakeholders: 
The insights gained will empower the following stakeholders to make informed strategic decisions and enhance manufacturing operations:
- Data analysts & business analysts
- Production managers
- Decision-makers & executives

### Business Questions:
- Why does the company fail to deliver these products on time?
- What strategies can be implemented to minimize scrapped products?
  

## II. Dataset Description

- Source: The Bicycle Manufacturer dataset is stored in a public Google BigQuery dataset named "adventureworks2019"
  To access the dataset, we log in to your Google Cloud Platform, navigate to the BigQuery console and search the project "adventureworks2019".
  
- Data Structure:
  There are 5 tables that we will work on it.

| Table                | Type                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Fact_Product         | Details of product sold or used in production.                                                                            |
| Fact_Workorder       | Contains order details, including product ID, scrapped products quantity, due date, start date, and end date.             |
| Dim_ScrapReason      | Reason for scrapped products.                                                                                             |
| Dim_WorkOrderRouting | Lists only on-time and late orders. Details of location, actual order and delivery time for each work order and product.  |
| Dim_Location         | Lists each stage in the production process.                                                                               |






## III. Exploring churn users' behaviour





## IV. ML model for predicting churned users 


## V. ML model for segmenting churned users

## VI. Final Conclusion & Recommendations 

The analysis has revealed some inefficiencies in manufacturing performance. The most critical problems are:



These issues directly impact manufacturing efficiency, delivery performance, and cost control, making them a priority for process improvement.

**Recommendations:**

