import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity



# Part C: Running a Factor Analysis
data = pd.read_csv("dataset_final.csv")

data = data.replace(0, np.nan)

data = data.dropna()

#print(data)

# Run a Factor Analysis

chi2, p = calculate_bartlett_sphericity(data)

#print(chi2, p)

machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(data)

ev, v = machine.get_eigenvalues()
#print(ev)

kaiser_criteria = ev[ev > 1]

print("Factors to Retain based on Kaiser's Criterion:", len(kaiser_criteria))

machine = FactorAnalyzer(n_factors=7, rotation=None)
machine.fit(data)
output = machine.loadings_
#print(output)

machine = FactorAnalyzer(n_factors=7, rotation='varimax')
machine.fit(data)
factor_loadings = machine.loadings_
#print(factor_loadings)

data = data.values

results = np.dot(data, factor_loadings)

pd.DataFrame(results).round().to_csv("results.csv", index=False)



