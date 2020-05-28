import pandas as pd
import matplotlib.pyplot as plt

sdata = pd.read_csv("skills.csv", skipinitialspace=True)
plt.style.use('ggplot')
plt.barh(y=sdata.skills, width=sdata.numbers)
plt.xticks(rotation=90)
plt.gca().invert_yaxis()