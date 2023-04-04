import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

years = np.array([2000, 2002, 2005, 2007, 2010]).reshape(-1, 1)
percentages = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

model = LinearRegression().fit(years, percentages)

def predict_unemployment(year):
    return model.predict(np.array([year]).reshape(-1, 1))[0]

year = 2010
while predict_unemployment(year) < 12:
    year += 1

plt.scatter(years, percentages, color='blue', label='Dane historyczne')
plt.plot(years, model.predict(years), color='red', label='Regresja liniowa')

plt.xlabel('Rok')
plt.ylabel('Procent bezrobotnych')
plt.title('Model regresji liniowej dla procentu bezrobotnych')

plt.legend()
plt.show()

print(f"Wynik modelu regresji liniowej dla roku 2015: {predict_unemployment(2015):.3f}")
print(f"Procent bezrobotnych przekroczy 12% w roku {year}.")