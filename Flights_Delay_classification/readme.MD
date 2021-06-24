# Data Science Take Home

Using the [U.S Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ)  between year 1999 - 2011 ...

# Tasks

1.  "... analyze the data and find out which factors & causes can lead to delays of flights?"
2.  "create a ML model that predicts whether a flight will be delayed."


# Exec. Summary

re.  **(1.)**, Factors that leads to delays:
-  _Departure Date_ (Day of the Month, Month, Day of Week)  
-  _Distance of flight_  

![features importance](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/features_importance_001.png?raw=true)

... are generally the top reasons (highest explanatory power toward the target variable, `late_flight`). More specifically, these attributes were derived from using data specific to an airline (American Eagle airline [OO])

As a double-check, it seems the same factors are also the top-most causes of delay when using data across all airports (whole year of 2007):
-  _Day of the Month_
-  _Distance of flight_



re. **(2)**, following are related files to models I've trained:

1. via a ML model trained for a _specific airline_:
	-   Source code: [Train_Model_airlinespecific.ipynb](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Train_Model_airlinespecific.ipynb)
	-   ML model: [Final_model___lightgbm___for_OO_07-05-2021 13:55:13.pkl](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Final_model___lightgbm___for_OO_07-05-2021%2013:55:13.pkl) (trained with American Eagle airline [OO])

1. via a ML model trained _across all airlines_:
	-   Source code: [Train_Model_general.ipynb](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Train_Model_general.ipynb)
	-   ML model: [Final_model___lightgbm___for_all_airlines_07-05-2021 13:31:37.pkl](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Final_model___lightgbm___for_all_airlines_07-05-2021%2013:31:37.pkl)



Lastly, you can find my **Data ETL and EDA** in
-   [Data_ETL_EDA_v1.ipynb](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Data_ETL_EDA_v1.ipynb)
-   [1_features_engr_v1.flow](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/1_features_engr_v1.flow)

NOTE, details of the 17-variables (features) utilized in training the model can be found below.


# Approach

### a.) Data ETL
- Ingest data to **AWS S3** as the primary storage of the raw and transformed data

- Transformed (cleaned) data two different methods (sytem)
  1. **AWS Sagemaker Data Wrangler** for data transformation and basic EDA
  2. Leverage **AWS Athena** to explore data structure before finalizing the transformation by creating [Data_ETL_EDA_v1.ipynb](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Data_ETL_EDA_v1.ipynb) and [1_features_engr_v1.ipynb](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/1_features_engr_v1.ipynb) --> [1_features_engr_v1.flow](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/1_features_engr_v1.flow)

I ended up using approach (2.) as I had more control and able to fine tune better.  But this could simply be the novice familiarity with Sagemaker Data Wrangler at the moment.

#### Data Volume
I thought I'd simply leverage the power of AWS to combine all years of data as a single dataset (except Sept-Nov 2011 due to a major/tragic event).  However, I later learnt that the data diversity is relatively homogenous across years.  Therefore, I ended up just using year 2007 data only.

#### Speed vs. Completeness
-   Even if using just 1-year of data (7M records), it took a significant amount of time and resources to get any indications of a model performance (using 4vCPU x16GiB x 1xGPU). As a result,
-   I'd decrease nFold value or omit cross-validation just so I can evaluate various scenarios (model with one airline, all airline, all airline for 1-month of data (Nov_2011)

### b.) ML Model Training
- AWS Sagemaker Studio for ML model training (Python)
- PyCaret framework to quickly get a sense of the effectiveness of different ML models.

I often use PyCaret to quickly get a sense of the effectiveness of different ML models. This way, I can quickly gauge a model's performance both statistically and its actionability by business users. However, i'd love to further investigate:
- data impute or imbalance due to multicollinearity (e.g., day of week / day of month versus traffic volume)
- features grouping
- models assembling/stacking

While Sagemaker and PyCaret 'automated' quite a bit of design and engineering of a traditional approach, and there were only 17-features used for the model; the train/validate/test dataset exploded to hundreds of dimension (e.g., one-hot encoding) due to the Categorical data type even when I was using _ml.g4dn.xlarge_ (4x Core GPU, 16MB RAM)

As a result, my approach ended up using Ultimately, the data set used were
- Decrease folds number or remove cross-validation
- Used 1-full year (2007): 7,294,649 rows --> Sampled 1% of data to train model (72,946 records)  
- For airline specific model (e.g., OO): 583,694 --> Sampled 12% of data to train model (70,043 records)

### c.) Model Deployment / Persistence
- AWS Sagemaker Studio for ML model training (Python)

Trained model were tested and validated within Sagemaker Studio environment as you can see from:
  - [Train_Model_airlinespecific.ipynb](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Train_Model_airlinespecific.ipynb)
  - [Train_Model_general.ipynb](https://github.com/rtheman/sages/blob/main/Flights_Delay_classification/Train_Model_general.ipynb)

Persistening the model simply by pickle-ing it with `save_model()` :-)
