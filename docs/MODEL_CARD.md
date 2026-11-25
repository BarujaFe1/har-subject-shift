# Model Card — HAR Under Subject Shift

**Purpose:** reproducible cross-subject benchmark and educational case study, not clinical decision support.

**Input:** 561 precomputed features from 2.56-second waist-mounted smartphone motion windows (UCI HAR). **Output:** one of six locomotion/posture classes.

**Model choice:** made on grouped nested CV, never on official holdout. Final trained estimator recorded in `artifacts/model_selection.json` only after actual execution.

**Generalization limits:** 30 scripted volunteers, single phone model and carrying position, 9 people in held-out split; does not validate other devices, populations, activities, sensors or medical applications. Sensitive potential behavioral patterns: do not attach subject IDs to predictors or publish raw subject trajectories.

**Training status in this archive:** NOT TRAINED ON OFFICIAL DATA. No test scores or model weights are represented as real.
