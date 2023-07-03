# venv OLS

# code comes from https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
import numpy as np
import pandas as pd
from linearmodels import PooledOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt


dataset = pd.read_excel("panel.xlsx", usecols=["region", "level_1", "x", "y"], index_col=[0, 1])
dataset["y"] = np.log(dataset["y"])
dataset.index.names = ["state", "year"]
x_lin = np.linspace(dataset["x"].min(), dataset["x"].max(), 100)

years = dataset.index.get_level_values("year").to_list()
dataset["year"] = pd.Categorical(years)


### plot ok
df = pd.read_excel("panel.xlsx", usecols=["region", "level_1", "x", "y"], index_col=[0, 1])
plt.plot(df["x"], df["y"])
###

# income is IV, for us it is x
# violent is DV, for us it is y

exog = sm.tools.tools.add_constant(dataset["x"])
endog = dataset["y"]
mod = PooledOLS(endog, exog)
pooledOLS_res = mod.fit(cov_type="clustered", cluster_entity=True)

# Store values for checking homoskedasticity graphically
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids


# 3A. Homoskedasticity


# 3A.1 Residuals-Plot for growing Variance Detection
fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color="blue")
ax.axhline(0, color="r", ls="--")
ax.set_xlabel("Predicted Values", fontsize=15)
ax.set_ylabel("Residuals", fontsize=15)
ax.set_title("Homoskedasticity Test", fontsize=30)
plt.show()


# 3A.2 White-Test
from statsmodels.stats.diagnostic import het_white, het_breuschpagan

pooled_OLS_dataset = pd.concat([dataset, residuals_pooled_OLS], axis=1)
pooled_OLS_dataset = pooled_OLS_dataset.drop(["year"], axis=1).fillna(0)
exog = sm.tools.tools.add_constant(dataset["x"]).fillna(0)
white_test_results = het_white(pooled_OLS_dataset["residual"], exog)
labels = ["LM-Stat", "LM p-val", "F-Stat", "F p-val"]
print(dict(zip(labels, white_test_results)))
# if p < 0.05, then heteroskedasticity is indicated
# here p=0.04 so heteroskedasticity
# error varies with predicted values


# 3A.3 Breusch-Pagan-Test
breusch_pagan_test_results = het_breuschpagan(pooled_OLS_dataset["residual"], exog)
labels = ["LM-Stat", "LM p-val", "F-Stat", "F p-val"]
print(dict(zip(labels, breusch_pagan_test_results)))
# p-value = 0.01


# 3.B Non-Autocorrelation
# Durbin-Watson-Test
from statsmodels.stats.stattools import durbin_watson

durbin_watson_test_results = durbin_watson(pooled_OLS_dataset["residual"])
print(durbin_watson_test_results)

# 0.87, positive autocorrelation

# FE und RE model
from linearmodels import PanelOLS
from linearmodels import RandomEffects

exog = sm.tools.tools.add_constant(dataset["x"])
endog = dataset["y"]
# random effects model
model_re = RandomEffects(endog, exog)
re_res = model_re.fit()
# fixed effects model
model_fe = PanelOLS(endog, exog, entity_effects=True)
fe_res = model_fe.fit()
# print results
print(re_res)
print(fe_res)


###
import numpy.linalg as la
from scipy import stats
import numpy as np


def hausman(fe, re):
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov
    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval


hausman_results = hausman(fe_res, re_res)
print("chi-Squared: " + str(hausman_results[0]))
print("degrees of freedom: " + str(hausman_results[1]))
print("p-Value: " + str(hausman_results[2]))

# test whether the individual characteristics are correlated with regressor
# null hypothesis is that they are not (random effect)
# here p-value = 0.0004 so fixed effect

fe_res.estimated_effects["estimated_effects"].unstack()[2015].round(2).sort_values().to_excel(
    "results/estimated_effects.xlsx"
)
