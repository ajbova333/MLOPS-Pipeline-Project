# Drift Monitoring Analysis

## Summary

I ran src/monitor_drift.py to compare the training data against a simulated production dataset. To build the production set, I intentionally shifted two features to immitate what a real population change might look like after model deployment. I picked age and added 5 years on average to the age and cholesterol adding 20 mg/dL on average. Those seemed to make sense as features that have an impact on potential heart health. They also both tend to change in tandem with each other as the model ages. I also added extra missing values to thalach and oldpeak to see how the report handled that. 

**Overall Drift Share: 14%(2 of 14 columns drifted)**

The features that difted were age and chol. The method used for monitoring was K-S test. Age drifted (p< 0.001) and chol did the same. All other features did not drift. 


## Which features showed drift and why?

As mentioned earlier in the summary age and chol were the only two features to drift. This is to be expected as those are the two features I shifted on purpose. Nothing else drifted including target which makes sense since the simulation did not touch the outcome labels. 

One thing worth noting I also added extra missing values into thalach and old peak. An additional 10% each on top of what was originally missing. Neither of those drifted. That isnt due to a bug its because DataDriftPreset only looks at the distribution of the values that are present not how many are missing. 


## Would this drift likely affect model performance?

More than likely yes, age and chol are both well-established risk factors for heart disease,
and both feed into the model directly as numeric inputs. If the patient population
shifts older and has higher cholesterol on average, the model starts seeing feature
values it saw less of during training — especially for chol, where a flat +20 mg/dL
shift pushes a chunk of patients into ranges that were rare in the training set. Even if
the actual relationship between these features and heart disease hasn't changed, the
model's decision boundary was fit to the old distribution, so it can start misclassifying
patients in the new one.

## What action would I recommend?

**Investigate first, retrain if it holds up.** A 14% drift share is under the 30%
threshold this script fails on, so it's not an emergency by itself — but age and chol
are two of the features I'd expect to matter most for this model, so I wouldn't just
pass this off either. Here's what I'd actually do:

1. Pull a bigger, more recent sample of production data and confirm this shift is real
   and sustained — not just noise from one batch.
2. Check the model's live accuracy/F1 against the thresholds set during training. If
   they've dropped, retrain on data that reflects the new population.
3. If performance is holding steady despite the drift, keep monitoring at the current
   cadence instead of retraining just because a number moved.