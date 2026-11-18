# Marketing email click prediction

Predict whether a user will click a laptop promotion email from their profile and browsing history. This Python coursework project combines user-level records with visit logs, builds behavior features, and fits a logistic regression classifier.

Author: Pengtian Sun  
Course: University of Wisconsin-Madison, COMP SCI 320, Machine Project 8: Selling Laptops - Smart Marketing

## Data and prediction target

The analysis uses six CSV files generated for the course exercise. The binary target is email clicking: 1 means clicked and 0 means did not click. It does not record a purchase. All six CSV files are included in the repository root; see DATA_PROVENANCE.md for their source and fields.

| Split | Users | Browsing log rows |
| --- | ---: | ---: |
| Training | 30,000 | 104,908 |
| Public test1 | 30,000 | 104,733 |

Each split has a users file, a logs file, and a labels file. Profiles include age, past purchase amount, a badge category, and a generated name. Logs associate a user with a visited URL and a duration value. The source field is called seconds; feature names follow that field. The assignment prose also refers to minutes, so duration units should be checked before making operational interpretations.

## Model workflow

1. Aggregate browsing records by user: total duration, visit count, average duration, maximum duration, and whether a laptop page was visited.
2. Join the aggregates to user profiles and fill behavior features with zero for users without browsing records.
3. Remove the user identifier from the predictors.
4. Impute and standardize numeric inputs; impute and one-hot encode categorical inputs.
5. Fit a scikit-learn Pipeline with LogisticRegression and apply the same preprocessing to test users.

The implementation keeps training and prediction feature construction in the same method. Its encoder can handle categories that were not seen during fitting.

## Evaluation

The evaluation script compares a majority-class baseline, the classifier with user profiles only, and the classifier with profiles plus browsing features. It reports accuracy, precision, recall, F1, and confusion matrices on the public test1 split. Labels are aligned by user ID, and the script checks for duplicate IDs, unknown log users, and train/test user overlap.

The reproducibility run on the public test1 split produced:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority-class baseline | 55.10% | 0.0000 | 0.0000 | 0.0000 |
| User profiles only | 64.88% | 0.6450 | 0.4844 | 0.5533 |
| User profiles and browsing | 80.28% | 0.7800 | 0.7811 | 0.7805 |

Adding browsing features increased accuracy by 15.40 percentage points over the profile-only classifier in this comparison. Full metrics and confusion matrices are recorded in evaluation.json. No hyperparameters are selected using test1 in this evaluation.

## Run locally

The evaluation was run with Python 3.12. Download or clone this repository, which includes train_users.csv, train_logs.csv, train_y.csv, test1_users.csv, test1_logs.csv, and test1_y.csv. From the repository directory, run:

~~~sh
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
python evaluate.py
~~~

To use CSV files stored elsewhere:

~~~sh
python evaluate.py --data-dir /path/to/csv-files --output evaluation.json
~~~

## Files

| File | Purpose |
| --- | --- |
| main.py | Coursework classifier and browsing-feature construction |
| evaluate.py | Portfolio evaluation script with baselines and a feature ablation |
| evaluation.json | Reproducible evaluation output, including software versions |
| requirements.txt | Versions of the main Python dependencies |
| DATA_PROVENANCE.md | Dataset source, schemas, and included data files |
| train_*.csv | Included course-generated training users, logs, and labels |
| test1_*.csv | Included course-generated public test users, logs, and labels |

## Interpretation and limitations

The data are generated for coursework, so these results do not establish real campaign performance or revenue impact. Test1 is a public course split and may have informed earlier coursework decisions; it is not a newly collected independent evaluation. This repository's evaluation uses the supplied classifier without a hyperparameter search and does not report cross-validation results.

The supplied model includes generated names as a categorical input. A further analysis should evaluate whether that feature adds stable signal, consider excluding it, and assess performance across user groups. Click prediction also does not establish that sending an email causes a click or a purchase.

## Attribution and scope

The assignment design and generated datasets come from UW-Madison COMP SCI 320 MP8. The classifier is preserved from the provided project implementation, with the course submission header replaced by a module description. The separate evaluation script and repository documentation are portfolio additions. Instructor grading scripts, answer keys, and submission instructions are excluded. No license is asserted for the underlying course materials.
