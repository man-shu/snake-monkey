# snake-monkey
Code for Snake-Monkey project

### Snake models
- BB: Bronze back
- CV: Cantor's pit viper
- PY: Python

### Variable definitions
- PD: Passing Distance (cm)
- PR: Proximity Distance (cm)
- FG: Frequency of Fear grimace (1/s)
- SS: Frequency of Self-scratching (1/s) 
- FR (= FG + SS): Frequency of Fear Response (1/s) 
- BS: Frequency of Bipedal Standing (1/s)
- GP: Gaze Percentage (%)
- FV: Frequency of Vocalisations (1/s)
- VO: Number of Vocalisations
- TT: Total time of interaction (s)

### Scripts
- `bb_py_cv.py` creates one boxplot for each variable, comparing it across the three snake models, with pair-wise statistical significance tests.
- `model_dead_live.py` creates one boxplot for each variable, comparing Cantor's pit viper presentation as a model, dead and live snake, with pair-wise statistical significance tests.
- `svm.py` performs a support vector machine classification on the behavioral data to predict the snake model (BB, CV, PY).
- `vocal_lda.py` performs a linear discriminant analysis on the vocalization data, to predict monkey calls.

### Dependencies
- Python 3.10.6
- seaborn 0.12.2
- numpy 1.25.1
- matplotlib 3.7.2
- scipy 1.11.1
- pandas 2.0.3
