import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

dataframes = {i: pd.read_csv(f"data/dataset_{i}.csv") for i in range(4)}

for i in range(4):
    x_mean = dataframes[i]['x'].mean()
    print(f"Dataset {i}: {x_mean=:.1f}")
print()

for i in range(4):
    y_mean = dataframes[i]['y'].mean()
    print(f"Dataset {i}: {y_mean=:.1f}")
print()

for i in range(4):
    x_std = dataframes[i]['x'].std()
    print(f"Dataset {i}: {x_std=:.1f}")
print()

for i in range(4):
    y_std = dataframes[i]['y'].std()
    print(f"Dataset {i}: {y_std=:.1f}")
print()

for i in range(4):
    corr_x_y = dataframes[i].corr().loc['x', 'y']
    print(f"Dataset {i}: {corr_x_y=:.1f}")
print()

for i in range(4):
    model = LinearRegression()
    x = dataframes[i]["x"].to_numpy().reshape(-1, 1)
    y = dataframes[i]["y"]
    model.fit(x, y)
    mean_squared_error = mse(model.predict(x), y)
    print(f"Dataset {i}: {mean_squared_error=:.1f}")
