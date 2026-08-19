# Project Progress & Decisions

## Overview
This document tracks the key steps, decisions, and findings throughout the project. Figures are stored in the `figures/` folder.

---

## Phase 1: Pre-processing

### Objective:

### Approach:

### Figures:

### Next Steps:

---

## Phase 2: Feature Extraction

### Objective:

### Approach:

### Figures:

### Next Steps:

---

## Phase 3: Data preperation

### Objective:
Prepare the data for modeling

### Approach:
I planned out the main steps I wanted do which included cleaning the data (dealing with outliers), formatting the data (encoding categorical data, scaling data, standardization and discretization) and potentially data engineering (whilst I would do this after having trained the first models). Before doing this I first had to split the data in train and test.

### Encoding categorical data
- I had two categorical data columns (category and valence) so encoded them using sklearn.preprocessing.OneHotEncoder

### Handling missing data
1. I first looked at how much data was missing per eye-coordinate recording (RDX, RDY, LDX, LDY)

![Barplot showing the percentage of missing data per eye-coordingate recording](figures/missing_pupil.png)

- I chose to focuse my analysis on only the RDX data as it had the least amount of missing data
2. I then looked at to see how much missing data there was per condition within the RDX data

![Barplot showing the percentage of missing data per condition in the RDX recordings](figures/missing_rdx.png)

3. I removed all of the rows where I had NaNs for the RDX data only. The table shows the effect of removing this data. The table shows the number of participants there was for each condition before and after removing the NaN rows.


|     | Before removing NaNs | After removing NaNs|
|:---:|:--------------------:|:------------------:|
|AMN  |          27          |          23        |
|DFT  |          24          |          20        |
|ACP  |          15          |          14        |
|CTR  |          21          |          17        |


### Train_test_split
- Before doing anything with the data I need to split it into train and test. GroupShuffleSplit was used to ensure that the same participant cannot be in both the train and test dataset. One issue is that I could get a test set with a condition being over or under represented. Need to look into whether I can fix this directly with GroupShuffleSplit or I need to use CV with k-fold iterations.


### Next Steps:
- Use GroupKFolds

