# COVID-19
COVID-19 Tracking in the USA from JHU (Johns Hopkins University) Data located at https://github.com/CSSEGISandData/COVID-19

## Prerequisites
Issues with downloading the CSV data via ssh can be corrected by installing/updating the **certifi** package
```
python3 -m pip install --upgrade certifi
```

## Helpful Hints
As more data is available, it may be necessary to update the initial conditions **p0** for the logistic and exponential model in order for the solver to properly calculate a fir for each model.