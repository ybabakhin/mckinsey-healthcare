# McKinsey Hackathon in Healthcare
It's a 15th place solution for McKinsey Analytics Hackathon in Healthcare:

https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon/

## Problem statement
The Client wants you to predict the probability of stroke happening to their patients. This will help doctors take proactive health measures for these patients.

## Solution
Solution is based on the blend of 2 simple Catboost models with the same set of features, but different parameters.

Due to high class imbalance, validation strategy was the major issue. Repeated stratified K-fold cross-validation was used. It gave pretty stable results with 0.8495 ROC AUC in Public LB and 0.8588 in the Private one, while there was tremendous shake-up in the LB

