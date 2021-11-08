import numpy as np
np.set_printoptions(threshold=1000,suppress=True)
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
from sklearn.pipeline import FeatureUnion

scroing=pd.read_csv('./Churn_ModellingTest.csv',sep=',',header=0) 

scroing = scroing.values
p = pickle.load(open('classifieur.pkl','rb'))
p.predict(scroing)