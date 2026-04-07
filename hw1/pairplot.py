import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")

df_sample = df.sample(5000)
sns.pairplot(df_sample)
plt.show()