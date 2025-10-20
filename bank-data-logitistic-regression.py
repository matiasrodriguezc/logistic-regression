#%%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


#%%Load the Data
raw_data = pd.read_csv('Bank_data.csv')
raw_data


#%%
# We make sure to create a copy of the data before we start altering it. Note that we don't change the original data we loaded.
data = raw_data.copy()
# Removes the index column thata comes with the data
data = data.drop(['Unnamed: 0'], axis = 1)
# We use the map function to change any 'yes' values to 1 and 'no'values to 0. 
data['y'] = data['y'].map({'yes':1, 'no':0})
data
data.describe()


#%%Declare the dependent and independent variables
y = data['y']
x1 = data['duration']


#%%Simple Logistic Regression
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()


#%%Interpretation
results_log.summary()

###The dependent variable is 'duration'. The model used is a Logit regression (logistic in common lingo), while the method - Maximum Likelihood Estimation (MLE). It has clearly converged after classifyin 518 observations. 
###The Pseudo R-squared is 0.21 which is within the 'acceptable region'. 
###The duration variable is significant and its coefficient is 0.0051.
###The constant is also significant and equals: -1.70