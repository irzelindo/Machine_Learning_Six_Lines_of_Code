import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labradors = 500
dogs = ['Greyhounds','Labradors']

grey_height = 28 + 4 * np.random.standard_normal(greyhounds)
lab_height = 24 + 4 * np.random.standard_normal(labradors)

plt.hist([grey_height, lab_height], stacked=True, color=['green', 'red'], rwidth=0.9, label=dogs)
plt.xlabel('Heights Distribution')
plt.xticks(list(range(16,38,2)))
plt.legend(prop={'size': 12})
plt.show()