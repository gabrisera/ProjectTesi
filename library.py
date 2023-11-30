# Importa le librerie necessarie
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import joblib
from sklearn.utils import resample
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from functions import normalize
from sklearn.metrics import make_scorer, f1_score