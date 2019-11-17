import pandas as pd

df = pd.read_pickle("./gcmt_all_earthquakes.pkl")

df = df[df.lat > 50]
df = df[df.lat < 75]
df = df[df.lon < -130]

print(df)

df.to_pickle("./gcmt_lat5075_lon130180.pkl")
