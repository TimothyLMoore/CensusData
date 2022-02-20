# Model Card

## Model Details

Creator: Timothy Moore

Model Date: February 8, 2022

Model Version: 1.0.0

Model Type: Random Forest Classifier

Citation Detail: Udacity Machine Learning DevOps Project #3

License: MIT License

Feedback: Please contact the model creator @ timothylmoore1985@gmail.com


## Intended Use

Intended use: The primary intended use is to determine somone's salary range based on limited information about them.

Intended user: Hobbyist, not intended to be used in any official capacity.

Out-of-scope uses: Never use this to determine someones salary.


## Training Data (80%)

A 32561 x 15 dataframe with the following columns.

* age
* workclass
* fnlgt
* education
* maritial status
* occupation
* relationship
* race
* sex
* capital-gain
* capital-loss
* hours-per-week
* native-counrty
* salary (target)


## Evaluation Data (20%)

The dataset was given by Udacity, it is publicly available Census Bureau data. Categorical data was One Hot Encoded and the target was passed through a Label Binarizer

## Metrics
Precision was the primary metric for the model, recall and F-beta scores are also reported.

The scores are as follows:
* Precision: 0.7998
* Recall: 0.5417
* F-Beta: 0.6459


## Ethical Considerations

While the data is anonymous it does contain sensitive data about class, occupation, race, country of origin, sex. The model should never be use to determine anything about anyone it is merely for learning purposes on my part. I went though and did slcies of the data based on "Race" and "Sex" and all the precision scores seemed reasonable.

## Caveats and Recommendations

I would have like to have more had data and more accurate data (the salary was limited to >50k and >=50k). The country of origin column had many American values, but too few other values to give any sort of reliable predictions after inference.
