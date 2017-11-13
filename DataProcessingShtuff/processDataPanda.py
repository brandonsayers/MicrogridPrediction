import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Import first data file.
# Set timestamp column as the index.
# Skip rows 1,2 because theyre useless.
# Normalize date-time format.
df1 = pd.read_csv('wind_data.csv', index_col=0, skiprows=[1,2])
df1.index = pd.to_datetime(df1.index)
print(df1)

# Import second data file.
df2 = pd.read_csv('powerstation.csv', parse_dates={'timestamp':[0,1]}, index_col=0)
print(df2)

# Merge both data sets using an outer join.
#	Note: Change to 'inner' for an inner join.
df3 = pd.merge(df2, df1, left_index=True, right_index=True, how='outer')
print(df3)

# Fill in all NaNs with 0s. 
# 	Note: Does not fill in "empty" cells.
df3.power_output = pd.to_numeric(df3.power_output, errors='coerce').fillna(0).astype(np.float64)
df3.Ts_Avg = pd.to_numeric(df3.Ts_Avg, errors='coerce').fillna(0).astype(np.float64)
df3.wnd_dir_compass = pd.to_numeric(df3.wnd_dir_compass, errors='coerce').fillna(0).astype(np.float64)
df3.wnd_spd = pd.to_numeric(df3.wnd_spd, errors='coerce').fillna(0).astype(np.float64)

# Export merged dataset to csv.
df3.to_csv('total.csv')

# Normalize data using sklearn library
x = df3.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df4 = pd.DataFrame(x_scaled)

df4.plot()
plt.show()

######## PROBABLY USELESS STUFF ##############
# df3.plot(x=df3.index, y=["power_output", "Ts_Avg", "wnd_dir_compass", "wnd_spd"])
# df3.plot(x=df3.index, y="power_output")
# df3.plot(x=df3.index, y="Ts_Avg")
# df3.plot(x=df3.index, y= "wnd_dir_compass")
# df3.plot(x=df3.index, y="wnd_spd")
# plt.savefig('figure.png')