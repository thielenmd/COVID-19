# COVID-19 Tracker for USA
# US data based on: https://github.com/CSSEGISandData/COVID-19

# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

CONST_SIGMA = 3                             # Smoothing value for gausian filter

# Define URL for data
# FIXED: https issue: python3 -m pip install --upgrade certifi
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

# Setup the datastructure for the CSV file
df = pd.read_csv(url, header=None)

# Pull the US data and Change the dates into numbers
# Take number of days since 1/1/2020
# US data is currently in the 226th row and dates start at column 4 (zero-origin)
df = df.iloc[[0,226],4:]
df = df.T
df.columns=["date", "total_cases"]
FMT = '%m/%d/%y'
date = df['date']
df['date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("1/1/20", FMT)).days )
df['total_cases'] = df['total_cases'].astype(int)

# Logistic Model - growth has an end
#   a = infection speed
#   b = day with the maximum infections occurred
#   c = total number of recorded infected people at the end of infection
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

# Curve fit to estimate parameter values and errors
x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
fit = curve_fit(logistic_model,x,y,p0=[5,200,2e6])
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]

a,b,c = fit[0][0],fit[0][1],fit[0][2]
end_date = c
print("a,b,c = ", a,b,c)
print("error(s) = ", errors)

# Find the infection end day
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
print("Infection End Day = Day ", sol)

# 95-99 percentile infection
sol_1 = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(0.9973*c),b))
print("99.7% (3-sigma) Projected Infections  = Day ", sol_1)

# Exponential Model - unstoppable growth
def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

exp_fit = curve_fit(exponential_model,x,y,p0=[2,2,200])

# Infection Rates
infection_rate = np.diff(y)

# Plots

pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [10, 8]

plt.rc('font', size=14)

# Real Data
plt.scatter(x,y,label="Real Data",color="red")

# Predicted End Date
plt.scatter(sol,end_date,label="Predicted End Date",color="blue")
plt.annotate(sol,(sol,end_date),textcoords="offset points",xytext=(0,-20),ha='center')

# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,a,b,c) for i in x+pred_x], label="Logistic Model")

# Predicted Exponential Curve
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential Model")

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.title("COVID-19 Cases in the United States",fontweight="bold")
plt.ylim((min(y)*0.9,c*1.1))

plt.grid(True)
plot_file = 'covid19_' + str(datetime.now().strftime('%Y_%m_%d')) + '.png'
plt.savefig(plot_file)
plt.show()

# Infection Rate Figure (Experimental) and filtered data
plt.figure()
plt.plot(x[1:],infection_rate,'.-',label="Real (Raw) Data",color="red")

rate_smoothed = gaussian_filter1d(infection_rate, sigma=CONST_SIGMA)
plt.plot(x[1:],rate_smoothed,label="Smoothed Data, $\sigma$ = " + str(CONST_SIGMA),color="blue")

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Infection Rate (Daily New Cases)")
plt.title("New COVID-19 Cases in the United States",fontweight="bold")

plt.grid(True)
plot_file = 'covid19_curve_' + str(datetime.now().strftime('%Y_%m_%d')) + '.png'
plt.savefig(plot_file)
plt.show()