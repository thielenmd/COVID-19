# COVID-19 Tracker for Italy

# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#%matplotlib inline                    # only used if iPython

# Define URL for data (US data will need transpose)
# Since pandas has trouble with https - download raw file and place
# in working directory
url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
local_file = "dpc-covid19-ita-andamento-nazionale.csv"

# Setup the datastructure for the CSV file
df = pd.read_csv(url)

# Change the dates into numbers
# Take number of days since 1/1/2020
df = df.loc[:,['data','totale_casi']]
FMT = '%Y-%m-%dT%H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01T00:00:00", FMT)).days )

# Logistic Model - growth has an end
#   a = infection speed
#   b = day with the maximum infections occurred
#   c = total number of recorded infected people at the end of infection
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

# Curve fit to estimate parameter values and errors
# fit is the curve fit to proivide a, b, c
# errors is the error in a, b, c
x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
fit = curve_fit(logistic_model,x,y,p0=[2,100,20000])
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]

a,b,c = fit[0][0],fit[0][1],fit[0][2]
print("a,b,c = ", a,b,c)

#Find the infection end day
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
print("Infection End Day = Day ", sol)

# Exponential Model - unstoppable growth
def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

exp_fit = curve_fit(exponential_model,x,y,p0=[1,5,20])

# Plots
pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [7, 7]

plt.rc('font', size=14)

# Real Data
plt.scatter(x,y,label="Real Data",color="red")

# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,a,b,c) for i in x+pred_x], label="Logistic Model")

# Predicted Exponential Curve
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential Model")

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))

plt.show()