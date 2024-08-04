# Binary_Classification_of_Insurance_Cross_Selling_Kaggle_Playground_Prediction_Competition-
# Overview
Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The objective of this competition is to predict which customers respond positively to an automobile insurance offer.

# Evaluation
Submissions are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.

Rank (610/2234)
<img src = 'https://github.com/anggapradanaa/Binary_Classification_of_Insurance_Cross_Selling_-Kaggle_Playground_Prediction_Competition-/blob/main/Leaderboard.png'>


# Importing Libraries and Dataset


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier
```


```python
# Load the dataset
df_train = pd.read_csv(r"C:\Users\ACER\Downloads\playground-series-s4e7\train.csv", index_col='id')
df_test = pd.read_csv(r"C:\Users\ACER\Downloads\playground-series-s4e7\test.csv", index_col='id')
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>21</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>65101.0</td>
      <td>124.0</td>
      <td>187</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>43</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>58911.0</td>
      <td>26.0</td>
      <td>288</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>25</td>
      <td>1</td>
      <td>14.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>38043.0</td>
      <td>152.0</td>
      <td>254</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>35</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>2630.0</td>
      <td>156.0</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>36</td>
      <td>1</td>
      <td>15.0</td>
      <td>1</td>
      <td>1-2 Year</td>
      <td>No</td>
      <td>31951.0</td>
      <td>152.0</td>
      <td>294</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Preprocessing and Data Splitting


```python
# Define function to convert data types
def converting_datatypes(df):
    df = df.copy()
    try:
        df['Gender'] = df['Gender'].astype('category')
        df['Vehicle_Age'] = df['Vehicle_Age'].astype('category')
        df['Vehicle_Damage'] = df['Vehicle_Damage'].astype('category')
        df['Age'] = df['Age'].astype('int8')
        df['Driving_License'] = df['Driving_License'].astype('category')
        df['Region_Code'] = df['Region_Code'].astype('int8')
        df['Previously_Insured'] = df['Previously_Insured'].astype('category')
        df['Annual_Premium'] = df['Annual_Premium'].astype('int32')
        df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('int16')
        df['Vintage'] = df['Vintage'].astype('int16')
        df['Response'] = df['Response'].astype('int8')
        print(df.info(memory_usage='deep'))
    except KeyError as e:
        print(f"Error: {e} not found in DataFrame")
    except Exception as e:
        print(f"An error occurred: {e}")
    return df
```


```python
df_train = converting_datatypes(df_train)
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 11504798 entries, 0 to 11504797
    Data columns (total 11 columns):
     #   Column                Dtype   
    ---  ------                -----   
     0   Gender                category
     1   Age                   int8    
     2   Driving_License       category
     3   Region_Code           int8    
     4   Previously_Insured    category
     5   Vehicle_Age           category
     6   Vehicle_Damage        category
     7   Annual_Premium        int32   
     8   Policy_Sales_Channel  int16   
     9   Vintage               int16   
     10  Response              int8    
    dtypes: category(5), int16(2), int32(1), int8(3)
    memory usage: 263.3 MB
    None
    


```python
# Define numerical and categorical pipelines
numerical_pipeline = Pipeline([
    ('scalar', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ('numeric', numerical_pipeline, ['Age',	'Region_Code',	'Annual_Premium',	'Policy_Sales_Channel',	'Vintage']),
    ('categoric', categorical_pipeline, ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage'])
])
```


```python
# Define the model pipeline
pipeline_catboost = Pipeline([
    ('prep', preprocessor),
    ('algo', CatBoostClassifier())
])
```


```python
# Split the training data
X = df_train.drop(columns='Response')
y = df_train['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Fit the Model


```python
# Fit the model
pipeline_catboost.fit(X_train, y_train)
```

    Learning rate set to 0.5
    0:	learn: 0.2987243	total: 1.2s	remaining: 19m 56s
    1:	learn: 0.2735083	total: 1.96s	remaining: 16m 18s
    2:	learn: 0.2685689	total: 2.68s	remaining: 14m 51s
    3:	learn: 0.2662971	total: 3.51s	remaining: 14m 33s
    4:	learn: 0.2652546	total: 4.2s	remaining: 13m 55s
    5:	learn: 0.2639522	total: 4.94s	remaining: 13m 38s
    6:	learn: 0.2630055	total: 5.68s	remaining: 13m 25s
    7:	learn: 0.2626634	total: 6.3s	remaining: 13m 1s
    8:	learn: 0.2621554	total: 7s	remaining: 12m 50s
    9:	learn: 0.2618846	total: 7.8s	remaining: 12m 51s
    10:	learn: 0.2616061	total: 8.5s	remaining: 12m 44s
    11:	learn: 0.2613648	total: 9.29s	remaining: 12m 45s
    12:	learn: 0.2611319	total: 10.1s	remaining: 12m 43s
    13:	learn: 0.2608972	total: 10.8s	remaining: 12m 40s
    14:	learn: 0.2606706	total: 11.5s	remaining: 12m 32s
    15:	learn: 0.2605314	total: 12.2s	remaining: 12m 32s
    16:	learn: 0.2603667	total: 13s	remaining: 12m 29s
    17:	learn: 0.2602490	total: 13.7s	remaining: 12m 25s
    18:	learn: 0.2600285	total: 14.5s	remaining: 12m 27s
    19:	learn: 0.2598233	total: 15.2s	remaining: 12m 25s
    20:	learn: 0.2596699	total: 15.9s	remaining: 12m 20s
    21:	learn: 0.2591619	total: 16.6s	remaining: 12m 19s
    22:	learn: 0.2591086	total: 17.2s	remaining: 12m 12s
    23:	learn: 0.2589684	total: 17.9s	remaining: 12m 7s
    24:	learn: 0.2588680	total: 18.6s	remaining: 12m 4s
    25:	learn: 0.2586665	total: 19.3s	remaining: 12m 1s
    26:	learn: 0.2585729	total: 20s	remaining: 12m 1s
    27:	learn: 0.2585196	total: 20.8s	remaining: 12m 2s
    28:	learn: 0.2581831	total: 21.6s	remaining: 12m 4s
    29:	learn: 0.2580179	total: 22.4s	remaining: 12m 5s
    30:	learn: 0.2578513	total: 23.1s	remaining: 12m 3s
    31:	learn: 0.2577835	total: 24s	remaining: 12m 5s
    32:	learn: 0.2576089	total: 24.7s	remaining: 12m 4s
    33:	learn: 0.2575207	total: 25.5s	remaining: 12m 3s
    34:	learn: 0.2573567	total: 26.3s	remaining: 12m 4s
    35:	learn: 0.2572393	total: 27s	remaining: 12m 1s
    36:	learn: 0.2571061	total: 27.7s	remaining: 12m
    37:	learn: 0.2570232	total: 28.4s	remaining: 11m 58s
    38:	learn: 0.2569370	total: 29s	remaining: 11m 55s
    39:	learn: 0.2568424	total: 29.8s	remaining: 11m 55s
    40:	learn: 0.2567764	total: 30.5s	remaining: 11m 54s
    41:	learn: 0.2565321	total: 31.3s	remaining: 11m 53s
    42:	learn: 0.2563735	total: 32.1s	remaining: 11m 54s
    43:	learn: 0.2561509	total: 32.8s	remaining: 11m 53s
    44:	learn: 0.2560199	total: 33.6s	remaining: 11m 53s
    45:	learn: 0.2559989	total: 34.4s	remaining: 11m 52s
    46:	learn: 0.2559253	total: 35.1s	remaining: 11m 51s
    47:	learn: 0.2558829	total: 35.8s	remaining: 11m 50s
    48:	learn: 0.2558091	total: 36.5s	remaining: 11m 48s
    49:	learn: 0.2556975	total: 37.2s	remaining: 11m 47s
    50:	learn: 0.2556513	total: 38s	remaining: 11m 47s
    51:	learn: 0.2555720	total: 38.7s	remaining: 11m 45s
    52:	learn: 0.2555180	total: 39.4s	remaining: 11m 44s
    53:	learn: 0.2554925	total: 40.2s	remaining: 11m 43s
    54:	learn: 0.2554438	total: 40.9s	remaining: 11m 42s
    55:	learn: 0.2553472	total: 41.6s	remaining: 11m 41s
    56:	learn: 0.2552703	total: 42.3s	remaining: 11m 40s
    57:	learn: 0.2552261	total: 43.1s	remaining: 11m 39s
    58:	learn: 0.2552017	total: 43.9s	remaining: 11m 39s
    59:	learn: 0.2551471	total: 44.6s	remaining: 11m 39s
    60:	learn: 0.2550931	total: 45.3s	remaining: 11m 37s
    61:	learn: 0.2550288	total: 46s	remaining: 11m 36s
    62:	learn: 0.2549697	total: 46.7s	remaining: 11m 34s
    63:	learn: 0.2549468	total: 47.3s	remaining: 11m 31s
    64:	learn: 0.2549211	total: 48.1s	remaining: 11m 31s
    65:	learn: 0.2548650	total: 48.8s	remaining: 11m 31s
    66:	learn: 0.2548027	total: 49.6s	remaining: 11m 30s
    67:	learn: 0.2547686	total: 50.3s	remaining: 11m 29s
    68:	learn: 0.2547466	total: 51s	remaining: 11m 28s
    69:	learn: 0.2547012	total: 51.8s	remaining: 11m 28s
    70:	learn: 0.2546777	total: 52.5s	remaining: 11m 26s
    71:	learn: 0.2546584	total: 53.2s	remaining: 11m 25s
    72:	learn: 0.2546105	total: 53.9s	remaining: 11m 24s
    73:	learn: 0.2545564	total: 54.7s	remaining: 11m 24s
    74:	learn: 0.2545293	total: 55.4s	remaining: 11m 23s
    75:	learn: 0.2544598	total: 56.2s	remaining: 11m 23s
    76:	learn: 0.2544409	total: 56.9s	remaining: 11m 21s
    77:	learn: 0.2543622	total: 57.6s	remaining: 11m 20s
    78:	learn: 0.2543051	total: 58.3s	remaining: 11m 19s
    79:	learn: 0.2542886	total: 59.1s	remaining: 11m 19s
    80:	learn: 0.2542799	total: 59.7s	remaining: 11m 17s
    81:	learn: 0.2542428	total: 1m	remaining: 11m 15s
    82:	learn: 0.2542136	total: 1m 1s	remaining: 11m 14s
    83:	learn: 0.2541883	total: 1m 1s	remaining: 11m 13s
    84:	learn: 0.2541759	total: 1m 2s	remaining: 11m 12s
    85:	learn: 0.2541374	total: 1m 3s	remaining: 11m 11s
    86:	learn: 0.2541142	total: 1m 3s	remaining: 11m 10s
    87:	learn: 0.2540953	total: 1m 4s	remaining: 11m 9s
    88:	learn: 0.2540579	total: 1m 5s	remaining: 11m 9s
    89:	learn: 0.2540202	total: 1m 6s	remaining: 11m 7s
    90:	learn: 0.2540008	total: 1m 6s	remaining: 11m 6s
    91:	learn: 0.2539154	total: 1m 7s	remaining: 11m 6s
    92:	learn: 0.2539030	total: 1m 8s	remaining: 11m 5s
    93:	learn: 0.2538791	total: 1m 9s	remaining: 11m 5s
    94:	learn: 0.2538618	total: 1m 9s	remaining: 11m 4s
    95:	learn: 0.2538245	total: 1m 10s	remaining: 11m 3s
    96:	learn: 0.2538088	total: 1m 11s	remaining: 11m 1s
    97:	learn: 0.2537762	total: 1m 11s	remaining: 11m
    98:	learn: 0.2537303	total: 1m 12s	remaining: 11m
    99:	learn: 0.2537213	total: 1m 13s	remaining: 10m 58s
    100:	learn: 0.2536874	total: 1m 13s	remaining: 10m 57s
    101:	learn: 0.2536622	total: 1m 14s	remaining: 10m 56s
    102:	learn: 0.2536244	total: 1m 15s	remaining: 10m 55s
    103:	learn: 0.2536089	total: 1m 16s	remaining: 10m 55s
    104:	learn: 0.2535938	total: 1m 16s	remaining: 10m 55s
    105:	learn: 0.2535530	total: 1m 17s	remaining: 10m 54s
    106:	learn: 0.2535288	total: 1m 18s	remaining: 10m 53s
    107:	learn: 0.2534782	total: 1m 18s	remaining: 10m 52s
    108:	learn: 0.2534430	total: 1m 19s	remaining: 10m 51s
    109:	learn: 0.2534002	total: 1m 20s	remaining: 10m 51s
    110:	learn: 0.2533743	total: 1m 21s	remaining: 10m 49s
    111:	learn: 0.2533583	total: 1m 21s	remaining: 10m 48s
    112:	learn: 0.2533486	total: 1m 22s	remaining: 10m 46s
    113:	learn: 0.2533386	total: 1m 23s	remaining: 10m 46s
    114:	learn: 0.2533191	total: 1m 23s	remaining: 10m 45s
    115:	learn: 0.2532967	total: 1m 24s	remaining: 10m 44s
    116:	learn: 0.2532779	total: 1m 25s	remaining: 10m 44s
    117:	learn: 0.2532592	total: 1m 26s	remaining: 10m 43s
    118:	learn: 0.2532468	total: 1m 26s	remaining: 10m 43s
    119:	learn: 0.2532360	total: 1m 27s	remaining: 10m 44s
    120:	learn: 0.2532296	total: 1m 28s	remaining: 10m 42s
    121:	learn: 0.2531679	total: 1m 29s	remaining: 10m 41s
    122:	learn: 0.2531393	total: 1m 29s	remaining: 10m 41s
    123:	learn: 0.2531042	total: 1m 30s	remaining: 10m 40s
    124:	learn: 0.2530863	total: 1m 31s	remaining: 10m 39s
    125:	learn: 0.2530625	total: 1m 32s	remaining: 10m 39s
    126:	learn: 0.2530468	total: 1m 32s	remaining: 10m 37s
    127:	learn: 0.2530369	total: 1m 33s	remaining: 10m 36s
    128:	learn: 0.2530246	total: 1m 34s	remaining: 10m 35s
    129:	learn: 0.2530126	total: 1m 34s	remaining: 10m 35s
    130:	learn: 0.2530056	total: 1m 35s	remaining: 10m 34s
    131:	learn: 0.2529829	total: 1m 36s	remaining: 10m 32s
    132:	learn: 0.2529564	total: 1m 36s	remaining: 10m 32s
    133:	learn: 0.2529394	total: 1m 37s	remaining: 10m 31s
    134:	learn: 0.2529285	total: 1m 38s	remaining: 10m 31s
    135:	learn: 0.2528861	total: 1m 39s	remaining: 10m 30s
    136:	learn: 0.2528398	total: 1m 39s	remaining: 10m 29s
    137:	learn: 0.2528198	total: 1m 40s	remaining: 10m 29s
    138:	learn: 0.2527840	total: 1m 41s	remaining: 10m 29s
    139:	learn: 0.2527560	total: 1m 42s	remaining: 10m 28s
    140:	learn: 0.2527284	total: 1m 43s	remaining: 10m 27s
    141:	learn: 0.2527038	total: 1m 43s	remaining: 10m 27s
    142:	learn: 0.2526831	total: 1m 44s	remaining: 10m 26s
    143:	learn: 0.2526557	total: 1m 45s	remaining: 10m 26s
    144:	learn: 0.2526461	total: 1m 46s	remaining: 10m 25s
    145:	learn: 0.2526311	total: 1m 46s	remaining: 10m 25s
    146:	learn: 0.2526142	total: 1m 47s	remaining: 10m 24s
    147:	learn: 0.2526001	total: 1m 48s	remaining: 10m 23s
    148:	learn: 0.2525952	total: 1m 48s	remaining: 10m 21s
    149:	learn: 0.2525888	total: 1m 49s	remaining: 10m 20s
    150:	learn: 0.2525709	total: 1m 50s	remaining: 10m 19s
    151:	learn: 0.2525558	total: 1m 51s	remaining: 10m 19s
    152:	learn: 0.2525502	total: 1m 51s	remaining: 10m 18s
    153:	learn: 0.2525348	total: 1m 52s	remaining: 10m 17s
    154:	learn: 0.2525205	total: 1m 53s	remaining: 10m 16s
    155:	learn: 0.2524872	total: 1m 53s	remaining: 10m 16s
    156:	learn: 0.2524727	total: 1m 54s	remaining: 10m 15s
    157:	learn: 0.2524610	total: 1m 55s	remaining: 10m 15s
    158:	learn: 0.2524574	total: 1m 56s	remaining: 10m 13s
    159:	learn: 0.2524433	total: 1m 56s	remaining: 10m 13s
    160:	learn: 0.2524314	total: 1m 57s	remaining: 10m 13s
    161:	learn: 0.2524186	total: 1m 58s	remaining: 10m 13s
    162:	learn: 0.2524071	total: 1m 59s	remaining: 10m 12s
    163:	learn: 0.2523979	total: 1m 59s	remaining: 10m 10s
    164:	learn: 0.2523870	total: 2m	remaining: 10m 9s
    165:	learn: 0.2523677	total: 2m 1s	remaining: 10m 9s
    166:	learn: 0.2523556	total: 2m 1s	remaining: 10m 7s
    167:	learn: 0.2523468	total: 2m 2s	remaining: 10m 6s
    168:	learn: 0.2523378	total: 2m 3s	remaining: 10m 5s
    169:	learn: 0.2523320	total: 2m 4s	remaining: 10m 5s
    170:	learn: 0.2523079	total: 2m 4s	remaining: 10m 4s
    171:	learn: 0.2522985	total: 2m 5s	remaining: 10m 4s
    172:	learn: 0.2522913	total: 2m 6s	remaining: 10m 3s
    173:	learn: 0.2522794	total: 2m 7s	remaining: 10m 2s
    174:	learn: 0.2522714	total: 2m 7s	remaining: 10m 2s
    175:	learn: 0.2522615	total: 2m 8s	remaining: 10m 2s
    176:	learn: 0.2522442	total: 2m 9s	remaining: 10m 1s
    177:	learn: 0.2522322	total: 2m 10s	remaining: 10m
    178:	learn: 0.2522235	total: 2m 10s	remaining: 10m
    179:	learn: 0.2522039	total: 2m 11s	remaining: 9m 58s
    180:	learn: 0.2521833	total: 2m 12s	remaining: 9m 57s
    181:	learn: 0.2521716	total: 2m 12s	remaining: 9m 56s
    182:	learn: 0.2521670	total: 2m 13s	remaining: 9m 55s
    183:	learn: 0.2521570	total: 2m 14s	remaining: 9m 54s
    184:	learn: 0.2521442	total: 2m 14s	remaining: 9m 53s
    185:	learn: 0.2521356	total: 2m 15s	remaining: 9m 53s
    186:	learn: 0.2521272	total: 2m 16s	remaining: 9m 52s
    187:	learn: 0.2521233	total: 2m 17s	remaining: 9m 51s
    188:	learn: 0.2521181	total: 2m 17s	remaining: 9m 50s
    189:	learn: 0.2521093	total: 2m 18s	remaining: 9m 49s
    190:	learn: 0.2520985	total: 2m 19s	remaining: 9m 48s
    191:	learn: 0.2520881	total: 2m 19s	remaining: 9m 48s
    192:	learn: 0.2520752	total: 2m 20s	remaining: 9m 48s
    193:	learn: 0.2520660	total: 2m 21s	remaining: 9m 47s
    194:	learn: 0.2520564	total: 2m 22s	remaining: 9m 46s
    195:	learn: 0.2520481	total: 2m 22s	remaining: 9m 45s
    196:	learn: 0.2520399	total: 2m 23s	remaining: 9m 45s
    197:	learn: 0.2520335	total: 2m 24s	remaining: 9m 44s
    198:	learn: 0.2520252	total: 2m 25s	remaining: 9m 43s
    199:	learn: 0.2520208	total: 2m 25s	remaining: 9m 42s
    200:	learn: 0.2520064	total: 2m 26s	remaining: 9m 42s
    201:	learn: 0.2519878	total: 2m 27s	remaining: 9m 41s
    202:	learn: 0.2519813	total: 2m 27s	remaining: 9m 40s
    203:	learn: 0.2519701	total: 2m 28s	remaining: 9m 39s
    204:	learn: 0.2519649	total: 2m 29s	remaining: 9m 38s
    205:	learn: 0.2519544	total: 2m 30s	remaining: 9m 38s
    206:	learn: 0.2519484	total: 2m 30s	remaining: 9m 37s
    207:	learn: 0.2519309	total: 2m 31s	remaining: 9m 36s
    208:	learn: 0.2519091	total: 2m 32s	remaining: 9m 35s
    209:	learn: 0.2519030	total: 2m 32s	remaining: 9m 34s
    210:	learn: 0.2518924	total: 2m 33s	remaining: 9m 34s
    211:	learn: 0.2518800	total: 2m 34s	remaining: 9m 33s
    212:	learn: 0.2518728	total: 2m 35s	remaining: 9m 33s
    213:	learn: 0.2518702	total: 2m 35s	remaining: 9m 32s
    214:	learn: 0.2518640	total: 2m 36s	remaining: 9m 30s
    215:	learn: 0.2518611	total: 2m 37s	remaining: 9m 30s
    216:	learn: 0.2518554	total: 2m 37s	remaining: 9m 29s
    217:	learn: 0.2518525	total: 2m 38s	remaining: 9m 28s
    218:	learn: 0.2518487	total: 2m 39s	remaining: 9m 28s
    219:	learn: 0.2518421	total: 2m 39s	remaining: 9m 27s
    220:	learn: 0.2518337	total: 2m 40s	remaining: 9m 26s
    221:	learn: 0.2518310	total: 2m 41s	remaining: 9m 26s
    222:	learn: 0.2518226	total: 2m 42s	remaining: 9m 25s
    223:	learn: 0.2518163	total: 2m 42s	remaining: 9m 24s
    224:	learn: 0.2518064	total: 2m 43s	remaining: 9m 24s
    225:	learn: 0.2517917	total: 2m 44s	remaining: 9m 23s
    226:	learn: 0.2517838	total: 2m 45s	remaining: 9m 22s
    227:	learn: 0.2517776	total: 2m 45s	remaining: 9m 21s
    228:	learn: 0.2517669	total: 2m 46s	remaining: 9m 20s
    229:	learn: 0.2517619	total: 2m 47s	remaining: 9m 19s
    230:	learn: 0.2517562	total: 2m 47s	remaining: 9m 18s
    231:	learn: 0.2517455	total: 2m 48s	remaining: 9m 17s
    232:	learn: 0.2517393	total: 2m 49s	remaining: 9m 16s
    233:	learn: 0.2517349	total: 2m 49s	remaining: 9m 16s
    234:	learn: 0.2517281	total: 2m 50s	remaining: 9m 15s
    235:	learn: 0.2517209	total: 2m 51s	remaining: 9m 14s
    236:	learn: 0.2517158	total: 2m 51s	remaining: 9m 13s
    237:	learn: 0.2517116	total: 2m 52s	remaining: 9m 12s
    238:	learn: 0.2517030	total: 2m 53s	remaining: 9m 12s
    239:	learn: 0.2516987	total: 2m 54s	remaining: 9m 11s
    240:	learn: 0.2516895	total: 2m 54s	remaining: 9m 10s
    241:	learn: 0.2516847	total: 2m 55s	remaining: 9m 9s
    242:	learn: 0.2516771	total: 2m 56s	remaining: 9m 8s
    243:	learn: 0.2516675	total: 2m 56s	remaining: 9m 8s
    244:	learn: 0.2516643	total: 2m 57s	remaining: 9m 7s
    245:	learn: 0.2516583	total: 2m 58s	remaining: 9m 6s
    246:	learn: 0.2516517	total: 2m 59s	remaining: 9m 6s
    247:	learn: 0.2516350	total: 2m 59s	remaining: 9m 5s
    248:	learn: 0.2516278	total: 3m	remaining: 9m 4s
    249:	learn: 0.2516217	total: 3m 1s	remaining: 9m 3s
    250:	learn: 0.2516134	total: 3m 1s	remaining: 9m 2s
    251:	learn: 0.2516010	total: 3m 2s	remaining: 9m 1s
    252:	learn: 0.2515970	total: 3m 3s	remaining: 9m 1s
    253:	learn: 0.2515848	total: 3m 4s	remaining: 9m
    254:	learn: 0.2515778	total: 3m 4s	remaining: 8m 59s
    255:	learn: 0.2515699	total: 3m 5s	remaining: 8m 59s
    256:	learn: 0.2515658	total: 3m 6s	remaining: 8m 58s
    257:	learn: 0.2515529	total: 3m 6s	remaining: 8m 57s
    258:	learn: 0.2515422	total: 3m 7s	remaining: 8m 56s
    259:	learn: 0.2515379	total: 3m 8s	remaining: 8m 55s
    260:	learn: 0.2515267	total: 3m 8s	remaining: 8m 55s
    261:	learn: 0.2515171	total: 3m 9s	remaining: 8m 54s
    262:	learn: 0.2515123	total: 3m 10s	remaining: 8m 53s
    263:	learn: 0.2515074	total: 3m 11s	remaining: 8m 53s
    264:	learn: 0.2515031	total: 3m 11s	remaining: 8m 52s
    265:	learn: 0.2514972	total: 3m 12s	remaining: 8m 51s
    266:	learn: 0.2514863	total: 3m 13s	remaining: 8m 50s
    267:	learn: 0.2514831	total: 3m 13s	remaining: 8m 49s
    268:	learn: 0.2514787	total: 3m 14s	remaining: 8m 48s
    269:	learn: 0.2514661	total: 3m 15s	remaining: 8m 47s
    270:	learn: 0.2514551	total: 3m 15s	remaining: 8m 47s
    271:	learn: 0.2514492	total: 3m 16s	remaining: 8m 46s
    272:	learn: 0.2514439	total: 3m 17s	remaining: 8m 45s
    273:	learn: 0.2514399	total: 3m 18s	remaining: 8m 45s
    274:	learn: 0.2514346	total: 3m 19s	remaining: 8m 44s
    275:	learn: 0.2514250	total: 3m 19s	remaining: 8m 43s
    276:	learn: 0.2514181	total: 3m 20s	remaining: 8m 43s
    277:	learn: 0.2514113	total: 3m 21s	remaining: 8m 42s
    278:	learn: 0.2514073	total: 3m 21s	remaining: 8m 41s
    279:	learn: 0.2513990	total: 3m 22s	remaining: 8m 41s
    280:	learn: 0.2513908	total: 3m 23s	remaining: 8m 40s
    281:	learn: 0.2513829	total: 3m 24s	remaining: 8m 39s
    282:	learn: 0.2513776	total: 3m 24s	remaining: 8m 39s
    283:	learn: 0.2513717	total: 3m 25s	remaining: 8m 38s
    284:	learn: 0.2513648	total: 3m 26s	remaining: 8m 37s
    285:	learn: 0.2513593	total: 3m 26s	remaining: 8m 36s
    286:	learn: 0.2513534	total: 3m 27s	remaining: 8m 36s
    287:	learn: 0.2513497	total: 3m 28s	remaining: 8m 35s
    288:	learn: 0.2513440	total: 3m 29s	remaining: 8m 34s
    289:	learn: 0.2513363	total: 3m 30s	remaining: 8m 34s
    290:	learn: 0.2513311	total: 3m 30s	remaining: 8m 33s
    291:	learn: 0.2513234	total: 3m 31s	remaining: 8m 32s
    292:	learn: 0.2513168	total: 3m 32s	remaining: 8m 32s
    293:	learn: 0.2513125	total: 3m 33s	remaining: 8m 32s
    294:	learn: 0.2513042	total: 3m 33s	remaining: 8m 31s
    295:	learn: 0.2512995	total: 3m 34s	remaining: 8m 30s
    296:	learn: 0.2512934	total: 3m 35s	remaining: 8m 29s
    297:	learn: 0.2512840	total: 3m 36s	remaining: 8m 29s
    298:	learn: 0.2512787	total: 3m 36s	remaining: 8m 28s
    299:	learn: 0.2512739	total: 3m 37s	remaining: 8m 28s
    300:	learn: 0.2512663	total: 3m 38s	remaining: 8m 27s
    301:	learn: 0.2512579	total: 3m 39s	remaining: 8m 26s
    302:	learn: 0.2512512	total: 3m 39s	remaining: 8m 26s
    303:	learn: 0.2512409	total: 3m 40s	remaining: 8m 25s
    304:	learn: 0.2512365	total: 3m 41s	remaining: 8m 24s
    305:	learn: 0.2512321	total: 3m 42s	remaining: 8m 23s
    306:	learn: 0.2512213	total: 3m 42s	remaining: 8m 23s
    307:	learn: 0.2512133	total: 3m 43s	remaining: 8m 22s
    308:	learn: 0.2512101	total: 3m 44s	remaining: 8m 21s
    309:	learn: 0.2512071	total: 3m 45s	remaining: 8m 21s
    310:	learn: 0.2512026	total: 3m 45s	remaining: 8m 20s
    311:	learn: 0.2511980	total: 3m 46s	remaining: 8m 19s
    312:	learn: 0.2511946	total: 3m 47s	remaining: 8m 18s
    313:	learn: 0.2511868	total: 3m 48s	remaining: 8m 18s
    314:	learn: 0.2511799	total: 3m 48s	remaining: 8m 17s
    315:	learn: 0.2511729	total: 3m 49s	remaining: 8m 16s
    316:	learn: 0.2511664	total: 3m 50s	remaining: 8m 16s
    317:	learn: 0.2511617	total: 3m 50s	remaining: 8m 15s
    318:	learn: 0.2511557	total: 3m 51s	remaining: 8m 14s
    319:	learn: 0.2511527	total: 3m 52s	remaining: 8m 13s
    320:	learn: 0.2511497	total: 3m 53s	remaining: 8m 12s
    321:	learn: 0.2511453	total: 3m 53s	remaining: 8m 12s
    322:	learn: 0.2511414	total: 3m 54s	remaining: 8m 11s
    323:	learn: 0.2511381	total: 3m 55s	remaining: 8m 10s
    324:	learn: 0.2511333	total: 3m 55s	remaining: 8m 10s
    325:	learn: 0.2511290	total: 3m 56s	remaining: 8m 9s
    326:	learn: 0.2511267	total: 3m 57s	remaining: 8m 8s
    327:	learn: 0.2511218	total: 3m 57s	remaining: 8m 7s
    328:	learn: 0.2511173	total: 3m 58s	remaining: 8m 6s
    329:	learn: 0.2511080	total: 3m 59s	remaining: 8m 6s
    330:	learn: 0.2511014	total: 4m	remaining: 8m 5s
    331:	learn: 0.2510944	total: 4m	remaining: 8m 4s
    332:	learn: 0.2510914	total: 4m 1s	remaining: 8m 3s
    333:	learn: 0.2510850	total: 4m 2s	remaining: 8m 2s
    334:	learn: 0.2510810	total: 4m 2s	remaining: 8m 1s
    335:	learn: 0.2510774	total: 4m 3s	remaining: 8m
    336:	learn: 0.2510732	total: 4m 4s	remaining: 8m
    337:	learn: 0.2510670	total: 4m 4s	remaining: 7m 59s
    338:	learn: 0.2510628	total: 4m 5s	remaining: 7m 58s
    339:	learn: 0.2510590	total: 4m 6s	remaining: 7m 57s
    340:	learn: 0.2510550	total: 4m 6s	remaining: 7m 57s
    341:	learn: 0.2510497	total: 4m 7s	remaining: 7m 56s
    342:	learn: 0.2510452	total: 4m 8s	remaining: 7m 55s
    343:	learn: 0.2510398	total: 4m 9s	remaining: 7m 54s
    344:	learn: 0.2510338	total: 4m 9s	remaining: 7m 54s
    345:	learn: 0.2510308	total: 4m 10s	remaining: 7m 53s
    346:	learn: 0.2510276	total: 4m 11s	remaining: 7m 53s
    347:	learn: 0.2510252	total: 4m 12s	remaining: 7m 52s
    348:	learn: 0.2510190	total: 4m 12s	remaining: 7m 51s
    349:	learn: 0.2510156	total: 4m 13s	remaining: 7m 50s
    350:	learn: 0.2510099	total: 4m 14s	remaining: 7m 50s
    351:	learn: 0.2510070	total: 4m 15s	remaining: 7m 49s
    352:	learn: 0.2509994	total: 4m 15s	remaining: 7m 48s
    353:	learn: 0.2509965	total: 4m 16s	remaining: 7m 48s
    354:	learn: 0.2509938	total: 4m 17s	remaining: 7m 47s
    355:	learn: 0.2509907	total: 4m 17s	remaining: 7m 46s
    356:	learn: 0.2509873	total: 4m 18s	remaining: 7m 46s
    357:	learn: 0.2509820	total: 4m 19s	remaining: 7m 45s
    358:	learn: 0.2509782	total: 4m 20s	remaining: 7m 44s
    359:	learn: 0.2509742	total: 4m 21s	remaining: 7m 44s
    360:	learn: 0.2509716	total: 4m 21s	remaining: 7m 43s
    361:	learn: 0.2509683	total: 4m 22s	remaining: 7m 42s
    362:	learn: 0.2509644	total: 4m 23s	remaining: 7m 42s
    363:	learn: 0.2509594	total: 4m 24s	remaining: 7m 41s
    364:	learn: 0.2509560	total: 4m 25s	remaining: 7m 41s
    365:	learn: 0.2509513	total: 4m 25s	remaining: 7m 40s
    366:	learn: 0.2509466	total: 4m 26s	remaining: 7m 39s
    367:	learn: 0.2509392	total: 4m 27s	remaining: 7m 39s
    368:	learn: 0.2509310	total: 4m 28s	remaining: 7m 38s
    369:	learn: 0.2509268	total: 4m 28s	remaining: 7m 37s
    370:	learn: 0.2509221	total: 4m 29s	remaining: 7m 36s
    371:	learn: 0.2509181	total: 4m 30s	remaining: 7m 36s
    372:	learn: 0.2509152	total: 4m 30s	remaining: 7m 35s
    373:	learn: 0.2509113	total: 4m 31s	remaining: 7m 34s
    374:	learn: 0.2509076	total: 4m 32s	remaining: 7m 33s
    375:	learn: 0.2509035	total: 4m 33s	remaining: 7m 33s
    376:	learn: 0.2508977	total: 4m 33s	remaining: 7m 32s
    377:	learn: 0.2508916	total: 4m 34s	remaining: 7m 31s
    378:	learn: 0.2508883	total: 4m 35s	remaining: 7m 31s
    379:	learn: 0.2508854	total: 4m 36s	remaining: 7m 30s
    380:	learn: 0.2508817	total: 4m 36s	remaining: 7m 29s
    381:	learn: 0.2508778	total: 4m 37s	remaining: 7m 29s
    382:	learn: 0.2508745	total: 4m 38s	remaining: 7m 28s
    383:	learn: 0.2508704	total: 4m 39s	remaining: 7m 27s
    384:	learn: 0.2508673	total: 4m 39s	remaining: 7m 26s
    385:	learn: 0.2508635	total: 4m 40s	remaining: 7m 26s
    386:	learn: 0.2508574	total: 4m 41s	remaining: 7m 25s
    387:	learn: 0.2508536	total: 4m 42s	remaining: 7m 25s
    388:	learn: 0.2508509	total: 4m 42s	remaining: 7m 24s
    389:	learn: 0.2508473	total: 4m 43s	remaining: 7m 23s
    390:	learn: 0.2508429	total: 4m 44s	remaining: 7m 22s
    391:	learn: 0.2508396	total: 4m 44s	remaining: 7m 21s
    392:	learn: 0.2508355	total: 4m 45s	remaining: 7m 21s
    393:	learn: 0.2508317	total: 4m 46s	remaining: 7m 20s
    394:	learn: 0.2508285	total: 4m 47s	remaining: 7m 19s
    395:	learn: 0.2508237	total: 4m 47s	remaining: 7m 19s
    396:	learn: 0.2508209	total: 4m 48s	remaining: 7m 18s
    397:	learn: 0.2508166	total: 4m 49s	remaining: 7m 17s
    398:	learn: 0.2508143	total: 4m 50s	remaining: 7m 17s
    399:	learn: 0.2508121	total: 4m 50s	remaining: 7m 16s
    400:	learn: 0.2508093	total: 4m 51s	remaining: 7m 15s
    401:	learn: 0.2508057	total: 4m 52s	remaining: 7m 14s
    402:	learn: 0.2508010	total: 4m 53s	remaining: 7m 14s
    403:	learn: 0.2507982	total: 4m 53s	remaining: 7m 13s
    404:	learn: 0.2507956	total: 4m 54s	remaining: 7m 12s
    405:	learn: 0.2507919	total: 4m 55s	remaining: 7m 11s
    406:	learn: 0.2507881	total: 4m 55s	remaining: 7m 11s
    407:	learn: 0.2507857	total: 4m 56s	remaining: 7m 10s
    408:	learn: 0.2507815	total: 4m 57s	remaining: 7m 9s
    409:	learn: 0.2507778	total: 4m 58s	remaining: 7m 9s
    410:	learn: 0.2507740	total: 4m 58s	remaining: 7m 8s
    411:	learn: 0.2507709	total: 4m 59s	remaining: 7m 7s
    412:	learn: 0.2507680	total: 5m	remaining: 7m 6s
    413:	learn: 0.2507645	total: 5m	remaining: 7m 6s
    414:	learn: 0.2507618	total: 5m 1s	remaining: 7m 5s
    415:	learn: 0.2507583	total: 5m 2s	remaining: 7m 4s
    416:	learn: 0.2507543	total: 5m 3s	remaining: 7m 3s
    417:	learn: 0.2507511	total: 5m 3s	remaining: 7m 3s
    418:	learn: 0.2507491	total: 5m 4s	remaining: 7m 2s
    419:	learn: 0.2507454	total: 5m 5s	remaining: 7m 1s
    420:	learn: 0.2507422	total: 5m 6s	remaining: 7m 1s
    421:	learn: 0.2507379	total: 5m 7s	remaining: 7m
    422:	learn: 0.2507356	total: 5m 7s	remaining: 6m 59s
    423:	learn: 0.2507320	total: 5m 8s	remaining: 6m 58s
    424:	learn: 0.2507295	total: 5m 9s	remaining: 6m 58s
    425:	learn: 0.2507269	total: 5m 9s	remaining: 6m 57s
    426:	learn: 0.2507244	total: 5m 10s	remaining: 6m 56s
    427:	learn: 0.2507186	total: 5m 11s	remaining: 6m 56s
    428:	learn: 0.2507140	total: 5m 12s	remaining: 6m 55s
    429:	learn: 0.2507112	total: 5m 12s	remaining: 6m 54s
    430:	learn: 0.2507081	total: 5m 13s	remaining: 6m 53s
    431:	learn: 0.2507026	total: 5m 14s	remaining: 6m 53s
    432:	learn: 0.2506975	total: 5m 14s	remaining: 6m 52s
    433:	learn: 0.2506950	total: 5m 15s	remaining: 6m 51s
    434:	learn: 0.2506918	total: 5m 16s	remaining: 6m 50s
    435:	learn: 0.2506853	total: 5m 17s	remaining: 6m 50s
    436:	learn: 0.2506802	total: 5m 17s	remaining: 6m 49s
    437:	learn: 0.2506768	total: 5m 18s	remaining: 6m 48s
    438:	learn: 0.2506737	total: 5m 19s	remaining: 6m 48s
    439:	learn: 0.2506705	total: 5m 20s	remaining: 6m 47s
    440:	learn: 0.2506661	total: 5m 20s	remaining: 6m 46s
    441:	learn: 0.2506628	total: 5m 21s	remaining: 6m 45s
    442:	learn: 0.2506591	total: 5m 22s	remaining: 6m 44s
    443:	learn: 0.2506573	total: 5m 22s	remaining: 6m 44s
    444:	learn: 0.2506519	total: 5m 23s	remaining: 6m 43s
    445:	learn: 0.2506480	total: 5m 24s	remaining: 6m 42s
    446:	learn: 0.2506448	total: 5m 24s	remaining: 6m 41s
    447:	learn: 0.2506423	total: 5m 25s	remaining: 6m 41s
    448:	learn: 0.2506376	total: 5m 26s	remaining: 6m 40s
    449:	learn: 0.2506346	total: 5m 27s	remaining: 6m 39s
    450:	learn: 0.2506294	total: 5m 27s	remaining: 6m 39s
    451:	learn: 0.2506270	total: 5m 28s	remaining: 6m 38s
    452:	learn: 0.2506248	total: 5m 29s	remaining: 6m 37s
    453:	learn: 0.2506218	total: 5m 30s	remaining: 6m 36s
    454:	learn: 0.2506194	total: 5m 30s	remaining: 6m 36s
    455:	learn: 0.2506122	total: 5m 31s	remaining: 6m 35s
    456:	learn: 0.2506089	total: 5m 32s	remaining: 6m 34s
    457:	learn: 0.2506065	total: 5m 33s	remaining: 6m 34s
    458:	learn: 0.2506022	total: 5m 33s	remaining: 6m 33s
    459:	learn: 0.2506002	total: 5m 34s	remaining: 6m 32s
    460:	learn: 0.2505961	total: 5m 35s	remaining: 6m 32s
    461:	learn: 0.2505931	total: 5m 36s	remaining: 6m 31s
    462:	learn: 0.2505894	total: 5m 36s	remaining: 6m 30s
    463:	learn: 0.2505861	total: 5m 37s	remaining: 6m 30s
    464:	learn: 0.2505826	total: 5m 38s	remaining: 6m 29s
    465:	learn: 0.2505797	total: 5m 39s	remaining: 6m 28s
    466:	learn: 0.2505769	total: 5m 39s	remaining: 6m 27s
    467:	learn: 0.2505734	total: 5m 40s	remaining: 6m 27s
    468:	learn: 0.2505701	total: 5m 41s	remaining: 6m 26s
    469:	learn: 0.2505677	total: 5m 41s	remaining: 6m 25s
    470:	learn: 0.2505644	total: 5m 42s	remaining: 6m 24s
    471:	learn: 0.2505617	total: 5m 43s	remaining: 6m 24s
    472:	learn: 0.2505554	total: 5m 43s	remaining: 6m 23s
    473:	learn: 0.2505532	total: 5m 44s	remaining: 6m 22s
    474:	learn: 0.2505448	total: 5m 45s	remaining: 6m 21s
    475:	learn: 0.2505424	total: 5m 46s	remaining: 6m 21s
    476:	learn: 0.2505408	total: 5m 46s	remaining: 6m 20s
    477:	learn: 0.2505382	total: 5m 47s	remaining: 6m 19s
    478:	learn: 0.2505354	total: 5m 48s	remaining: 6m 18s
    479:	learn: 0.2505311	total: 5m 49s	remaining: 6m 18s
    480:	learn: 0.2505285	total: 5m 49s	remaining: 6m 17s
    481:	learn: 0.2505259	total: 5m 50s	remaining: 6m 16s
    482:	learn: 0.2505234	total: 5m 51s	remaining: 6m 15s
    483:	learn: 0.2505206	total: 5m 51s	remaining: 6m 15s
    484:	learn: 0.2505183	total: 5m 52s	remaining: 6m 14s
    485:	learn: 0.2505153	total: 5m 53s	remaining: 6m 13s
    486:	learn: 0.2505119	total: 5m 54s	remaining: 6m 13s
    487:	learn: 0.2505071	total: 5m 54s	remaining: 6m 12s
    488:	learn: 0.2505048	total: 5m 55s	remaining: 6m 11s
    489:	learn: 0.2505019	total: 5m 56s	remaining: 6m 10s
    490:	learn: 0.2504982	total: 5m 57s	remaining: 6m 10s
    491:	learn: 0.2504954	total: 5m 57s	remaining: 6m 9s
    492:	learn: 0.2504922	total: 5m 58s	remaining: 6m 8s
    493:	learn: 0.2504897	total: 5m 59s	remaining: 6m 8s
    494:	learn: 0.2504856	total: 6m	remaining: 6m 7s
    495:	learn: 0.2504792	total: 6m	remaining: 6m 6s
    496:	learn: 0.2504769	total: 6m 1s	remaining: 6m 5s
    497:	learn: 0.2504714	total: 6m 2s	remaining: 6m 5s
    498:	learn: 0.2504690	total: 6m 3s	remaining: 6m 4s
    499:	learn: 0.2504664	total: 6m 3s	remaining: 6m 3s
    500:	learn: 0.2504647	total: 6m 4s	remaining: 6m 3s
    501:	learn: 0.2504603	total: 6m 5s	remaining: 6m 2s
    502:	learn: 0.2504559	total: 6m 6s	remaining: 6m 1s
    503:	learn: 0.2504505	total: 6m 6s	remaining: 6m 1s
    504:	learn: 0.2504471	total: 6m 7s	remaining: 6m
    505:	learn: 0.2504435	total: 6m 8s	remaining: 5m 59s
    506:	learn: 0.2504390	total: 6m 9s	remaining: 5m 58s
    507:	learn: 0.2504354	total: 6m 9s	remaining: 5m 58s
    508:	learn: 0.2504320	total: 6m 10s	remaining: 5m 57s
    509:	learn: 0.2504294	total: 6m 11s	remaining: 5m 56s
    510:	learn: 0.2504261	total: 6m 12s	remaining: 5m 56s
    511:	learn: 0.2504205	total: 6m 12s	remaining: 5m 55s
    512:	learn: 0.2504172	total: 6m 13s	remaining: 5m 54s
    513:	learn: 0.2504140	total: 6m 14s	remaining: 5m 53s
    514:	learn: 0.2504094	total: 6m 14s	remaining: 5m 53s
    515:	learn: 0.2504057	total: 6m 15s	remaining: 5m 52s
    516:	learn: 0.2504015	total: 6m 16s	remaining: 5m 51s
    517:	learn: 0.2503991	total: 6m 16s	remaining: 5m 50s
    518:	learn: 0.2503965	total: 6m 17s	remaining: 5m 49s
    519:	learn: 0.2503934	total: 6m 18s	remaining: 5m 49s
    520:	learn: 0.2503907	total: 6m 18s	remaining: 5m 48s
    521:	learn: 0.2503853	total: 6m 19s	remaining: 5m 47s
    522:	learn: 0.2503817	total: 6m 20s	remaining: 5m 46s
    523:	learn: 0.2503786	total: 6m 21s	remaining: 5m 46s
    524:	learn: 0.2503745	total: 6m 22s	remaining: 5m 45s
    525:	learn: 0.2503710	total: 6m 22s	remaining: 5m 44s
    526:	learn: 0.2503674	total: 6m 23s	remaining: 5m 44s
    527:	learn: 0.2503625	total: 6m 24s	remaining: 5m 43s
    528:	learn: 0.2503592	total: 6m 25s	remaining: 5m 42s
    529:	learn: 0.2503571	total: 6m 25s	remaining: 5m 42s
    530:	learn: 0.2503547	total: 6m 26s	remaining: 5m 41s
    531:	learn: 0.2503510	total: 6m 27s	remaining: 5m 40s
    532:	learn: 0.2503472	total: 6m 27s	remaining: 5m 39s
    533:	learn: 0.2503426	total: 6m 28s	remaining: 5m 39s
    534:	learn: 0.2503405	total: 6m 29s	remaining: 5m 38s
    535:	learn: 0.2503379	total: 6m 30s	remaining: 5m 37s
    536:	learn: 0.2503344	total: 6m 31s	remaining: 5m 37s
    537:	learn: 0.2503315	total: 6m 31s	remaining: 5m 36s
    538:	learn: 0.2503284	total: 6m 32s	remaining: 5m 35s
    539:	learn: 0.2503255	total: 6m 33s	remaining: 5m 34s
    540:	learn: 0.2503225	total: 6m 33s	remaining: 5m 34s
    541:	learn: 0.2503189	total: 6m 34s	remaining: 5m 33s
    542:	learn: 0.2503163	total: 6m 35s	remaining: 5m 32s
    543:	learn: 0.2503139	total: 6m 36s	remaining: 5m 32s
    544:	learn: 0.2503102	total: 6m 36s	remaining: 5m 31s
    545:	learn: 0.2503073	total: 6m 37s	remaining: 5m 30s
    546:	learn: 0.2503039	total: 6m 38s	remaining: 5m 30s
    547:	learn: 0.2503009	total: 6m 39s	remaining: 5m 29s
    548:	learn: 0.2502978	total: 6m 39s	remaining: 5m 28s
    549:	learn: 0.2502956	total: 6m 40s	remaining: 5m 27s
    550:	learn: 0.2502936	total: 6m 41s	remaining: 5m 26s
    551:	learn: 0.2502899	total: 6m 42s	remaining: 5m 26s
    552:	learn: 0.2502864	total: 6m 42s	remaining: 5m 25s
    553:	learn: 0.2502828	total: 6m 43s	remaining: 5m 24s
    554:	learn: 0.2502803	total: 6m 44s	remaining: 5m 24s
    555:	learn: 0.2502780	total: 6m 45s	remaining: 5m 23s
    556:	learn: 0.2502747	total: 6m 45s	remaining: 5m 22s
    557:	learn: 0.2502714	total: 6m 46s	remaining: 5m 21s
    558:	learn: 0.2502687	total: 6m 47s	remaining: 5m 21s
    559:	learn: 0.2502660	total: 6m 47s	remaining: 5m 20s
    560:	learn: 0.2502635	total: 6m 48s	remaining: 5m 19s
    561:	learn: 0.2502605	total: 6m 49s	remaining: 5m 19s
    562:	learn: 0.2502568	total: 6m 50s	remaining: 5m 18s
    563:	learn: 0.2502539	total: 6m 50s	remaining: 5m 17s
    564:	learn: 0.2502502	total: 6m 51s	remaining: 5m 16s
    565:	learn: 0.2502481	total: 6m 52s	remaining: 5m 16s
    566:	learn: 0.2502449	total: 6m 53s	remaining: 5m 15s
    567:	learn: 0.2502406	total: 6m 54s	remaining: 5m 14s
    568:	learn: 0.2502369	total: 6m 54s	remaining: 5m 14s
    569:	learn: 0.2502352	total: 6m 55s	remaining: 5m 13s
    570:	learn: 0.2502302	total: 6m 56s	remaining: 5m 12s
    571:	learn: 0.2502258	total: 6m 57s	remaining: 5m 12s
    572:	learn: 0.2502234	total: 6m 57s	remaining: 5m 11s
    573:	learn: 0.2502197	total: 6m 58s	remaining: 5m 10s
    574:	learn: 0.2502163	total: 6m 59s	remaining: 5m 9s
    575:	learn: 0.2502112	total: 7m	remaining: 5m 9s
    576:	learn: 0.2502074	total: 7m	remaining: 5m 8s
    577:	learn: 0.2502047	total: 7m 1s	remaining: 5m 7s
    578:	learn: 0.2502027	total: 7m 2s	remaining: 5m 7s
    579:	learn: 0.2502002	total: 7m 2s	remaining: 5m 6s
    580:	learn: 0.2501947	total: 7m 3s	remaining: 5m 5s
    581:	learn: 0.2501909	total: 7m 4s	remaining: 5m 4s
    582:	learn: 0.2501883	total: 7m 5s	remaining: 5m 4s
    583:	learn: 0.2501852	total: 7m 5s	remaining: 5m 3s
    584:	learn: 0.2501824	total: 7m 6s	remaining: 5m 2s
    585:	learn: 0.2501798	total: 7m 7s	remaining: 5m 1s
    586:	learn: 0.2501764	total: 7m 8s	remaining: 5m 1s
    587:	learn: 0.2501736	total: 7m 8s	remaining: 5m
    588:	learn: 0.2501698	total: 7m 9s	remaining: 4m 59s
    589:	learn: 0.2501672	total: 7m 10s	remaining: 4m 59s
    590:	learn: 0.2501647	total: 7m 11s	remaining: 4m 58s
    591:	learn: 0.2501624	total: 7m 12s	remaining: 4m 57s
    592:	learn: 0.2501587	total: 7m 12s	remaining: 4m 57s
    593:	learn: 0.2501563	total: 7m 13s	remaining: 4m 56s
    594:	learn: 0.2501542	total: 7m 14s	remaining: 4m 55s
    595:	learn: 0.2501518	total: 7m 15s	remaining: 4m 54s
    596:	learn: 0.2501489	total: 7m 15s	remaining: 4m 54s
    597:	learn: 0.2501471	total: 7m 16s	remaining: 4m 53s
    598:	learn: 0.2501431	total: 7m 17s	remaining: 4m 52s
    599:	learn: 0.2501397	total: 7m 17s	remaining: 4m 51s
    600:	learn: 0.2501374	total: 7m 18s	remaining: 4m 50s
    601:	learn: 0.2501347	total: 7m 19s	remaining: 4m 50s
    602:	learn: 0.2501320	total: 7m 19s	remaining: 4m 49s
    603:	learn: 0.2501291	total: 7m 20s	remaining: 4m 48s
    604:	learn: 0.2501269	total: 7m 21s	remaining: 4m 48s
    605:	learn: 0.2501252	total: 7m 22s	remaining: 4m 47s
    606:	learn: 0.2501218	total: 7m 23s	remaining: 4m 46s
    607:	learn: 0.2501189	total: 7m 23s	remaining: 4m 46s
    608:	learn: 0.2501164	total: 7m 24s	remaining: 4m 45s
    609:	learn: 0.2501143	total: 7m 25s	remaining: 4m 44s
    610:	learn: 0.2501117	total: 7m 26s	remaining: 4m 43s
    611:	learn: 0.2501087	total: 7m 26s	remaining: 4m 43s
    612:	learn: 0.2501066	total: 7m 27s	remaining: 4m 42s
    613:	learn: 0.2501041	total: 7m 28s	remaining: 4m 41s
    614:	learn: 0.2501016	total: 7m 29s	remaining: 4m 41s
    615:	learn: 0.2500990	total: 7m 29s	remaining: 4m 40s
    616:	learn: 0.2500965	total: 7m 30s	remaining: 4m 39s
    617:	learn: 0.2500935	total: 7m 31s	remaining: 4m 38s
    618:	learn: 0.2500911	total: 7m 32s	remaining: 4m 38s
    619:	learn: 0.2500888	total: 7m 32s	remaining: 4m 37s
    620:	learn: 0.2500868	total: 7m 33s	remaining: 4m 36s
    621:	learn: 0.2500849	total: 7m 34s	remaining: 4m 35s
    622:	learn: 0.2500809	total: 7m 34s	remaining: 4m 35s
    623:	learn: 0.2500780	total: 7m 35s	remaining: 4m 34s
    624:	learn: 0.2500759	total: 7m 36s	remaining: 4m 33s
    625:	learn: 0.2500730	total: 7m 36s	remaining: 4m 32s
    626:	learn: 0.2500705	total: 7m 37s	remaining: 4m 32s
    627:	learn: 0.2500687	total: 7m 38s	remaining: 4m 31s
    628:	learn: 0.2500671	total: 7m 39s	remaining: 4m 30s
    629:	learn: 0.2500655	total: 7m 39s	remaining: 4m 30s
    630:	learn: 0.2500630	total: 7m 40s	remaining: 4m 29s
    631:	learn: 0.2500605	total: 7m 41s	remaining: 4m 28s
    632:	learn: 0.2500585	total: 7m 42s	remaining: 4m 28s
    633:	learn: 0.2500553	total: 7m 43s	remaining: 4m 27s
    634:	learn: 0.2500524	total: 7m 43s	remaining: 4m 26s
    635:	learn: 0.2500498	total: 7m 44s	remaining: 4m 25s
    636:	learn: 0.2500489	total: 7m 45s	remaining: 4m 25s
    637:	learn: 0.2500467	total: 7m 46s	remaining: 4m 24s
    638:	learn: 0.2500444	total: 7m 46s	remaining: 4m 23s
    639:	learn: 0.2500417	total: 7m 47s	remaining: 4m 23s
    640:	learn: 0.2500389	total: 7m 48s	remaining: 4m 22s
    641:	learn: 0.2500366	total: 7m 49s	remaining: 4m 21s
    642:	learn: 0.2500343	total: 7m 49s	remaining: 4m 20s
    643:	learn: 0.2500304	total: 7m 50s	remaining: 4m 20s
    644:	learn: 0.2500289	total: 7m 51s	remaining: 4m 19s
    645:	learn: 0.2500267	total: 7m 52s	remaining: 4m 18s
    646:	learn: 0.2500243	total: 7m 52s	remaining: 4m 17s
    647:	learn: 0.2500212	total: 7m 53s	remaining: 4m 17s
    648:	learn: 0.2500188	total: 7m 54s	remaining: 4m 16s
    649:	learn: 0.2500170	total: 7m 54s	remaining: 4m 15s
    650:	learn: 0.2500153	total: 7m 55s	remaining: 4m 14s
    651:	learn: 0.2500125	total: 7m 56s	remaining: 4m 14s
    652:	learn: 0.2500105	total: 7m 57s	remaining: 4m 13s
    653:	learn: 0.2500084	total: 7m 57s	remaining: 4m 12s
    654:	learn: 0.2500065	total: 7m 58s	remaining: 4m 12s
    655:	learn: 0.2500043	total: 7m 59s	remaining: 4m 11s
    656:	learn: 0.2500007	total: 8m	remaining: 4m 10s
    657:	learn: 0.2499985	total: 8m	remaining: 4m 9s
    658:	learn: 0.2499951	total: 8m 1s	remaining: 4m 9s
    659:	learn: 0.2499925	total: 8m 2s	remaining: 4m 8s
    660:	learn: 0.2499909	total: 8m 2s	remaining: 4m 7s
    661:	learn: 0.2499886	total: 8m 3s	remaining: 4m 6s
    662:	learn: 0.2499867	total: 8m 4s	remaining: 4m 6s
    663:	learn: 0.2499846	total: 8m 5s	remaining: 4m 5s
    664:	learn: 0.2499818	total: 8m 5s	remaining: 4m 4s
    665:	learn: 0.2499799	total: 8m 6s	remaining: 4m 4s
    666:	learn: 0.2499774	total: 8m 7s	remaining: 4m 3s
    667:	learn: 0.2499740	total: 8m 8s	remaining: 4m 2s
    668:	learn: 0.2499718	total: 8m 8s	remaining: 4m 1s
    669:	learn: 0.2499688	total: 8m 9s	remaining: 4m 1s
    670:	learn: 0.2499654	total: 8m 10s	remaining: 4m
    671:	learn: 0.2499629	total: 8m 10s	remaining: 3m 59s
    672:	learn: 0.2499611	total: 8m 11s	remaining: 3m 58s
    673:	learn: 0.2499592	total: 8m 12s	remaining: 3m 58s
    674:	learn: 0.2499570	total: 8m 13s	remaining: 3m 57s
    675:	learn: 0.2499514	total: 8m 13s	remaining: 3m 56s
    676:	learn: 0.2499476	total: 8m 14s	remaining: 3m 56s
    677:	learn: 0.2499448	total: 8m 15s	remaining: 3m 55s
    678:	learn: 0.2499431	total: 8m 16s	remaining: 3m 54s
    679:	learn: 0.2499405	total: 8m 16s	remaining: 3m 53s
    680:	learn: 0.2499375	total: 8m 17s	remaining: 3m 53s
    681:	learn: 0.2499354	total: 8m 18s	remaining: 3m 52s
    682:	learn: 0.2499332	total: 8m 18s	remaining: 3m 51s
    683:	learn: 0.2499298	total: 8m 19s	remaining: 3m 50s
    684:	learn: 0.2499279	total: 8m 20s	remaining: 3m 50s
    685:	learn: 0.2499252	total: 8m 21s	remaining: 3m 49s
    686:	learn: 0.2499234	total: 8m 21s	remaining: 3m 48s
    687:	learn: 0.2499204	total: 8m 22s	remaining: 3m 48s
    688:	learn: 0.2499180	total: 8m 23s	remaining: 3m 47s
    689:	learn: 0.2499161	total: 8m 24s	remaining: 3m 46s
    690:	learn: 0.2499143	total: 8m 25s	remaining: 3m 45s
    691:	learn: 0.2499127	total: 8m 25s	remaining: 3m 45s
    692:	learn: 0.2499110	total: 8m 26s	remaining: 3m 44s
    693:	learn: 0.2499087	total: 8m 27s	remaining: 3m 43s
    694:	learn: 0.2499058	total: 8m 28s	remaining: 3m 42s
    695:	learn: 0.2499037	total: 8m 28s	remaining: 3m 42s
    696:	learn: 0.2499011	total: 8m 29s	remaining: 3m 41s
    697:	learn: 0.2498982	total: 8m 30s	remaining: 3m 40s
    698:	learn: 0.2498968	total: 8m 31s	remaining: 3m 40s
    699:	learn: 0.2498952	total: 8m 31s	remaining: 3m 39s
    700:	learn: 0.2498928	total: 8m 32s	remaining: 3m 38s
    701:	learn: 0.2498889	total: 8m 33s	remaining: 3m 37s
    702:	learn: 0.2498863	total: 8m 33s	remaining: 3m 37s
    703:	learn: 0.2498844	total: 8m 34s	remaining: 3m 36s
    704:	learn: 0.2498819	total: 8m 35s	remaining: 3m 35s
    705:	learn: 0.2498800	total: 8m 35s	remaining: 3m 34s
    706:	learn: 0.2498772	total: 8m 36s	remaining: 3m 34s
    707:	learn: 0.2498750	total: 8m 37s	remaining: 3m 33s
    708:	learn: 0.2498719	total: 8m 38s	remaining: 3m 32s
    709:	learn: 0.2498695	total: 8m 38s	remaining: 3m 31s
    710:	learn: 0.2498673	total: 8m 39s	remaining: 3m 31s
    711:	learn: 0.2498656	total: 8m 40s	remaining: 3m 30s
    712:	learn: 0.2498622	total: 8m 41s	remaining: 3m 29s
    713:	learn: 0.2498583	total: 8m 41s	remaining: 3m 28s
    714:	learn: 0.2498557	total: 8m 42s	remaining: 3m 28s
    715:	learn: 0.2498520	total: 8m 43s	remaining: 3m 27s
    716:	learn: 0.2498491	total: 8m 43s	remaining: 3m 26s
    717:	learn: 0.2498451	total: 8m 44s	remaining: 3m 26s
    718:	learn: 0.2498429	total: 8m 45s	remaining: 3m 25s
    719:	learn: 0.2498418	total: 8m 45s	remaining: 3m 24s
    720:	learn: 0.2498394	total: 8m 46s	remaining: 3m 23s
    721:	learn: 0.2498368	total: 8m 47s	remaining: 3m 23s
    722:	learn: 0.2498344	total: 8m 47s	remaining: 3m 22s
    723:	learn: 0.2498323	total: 8m 48s	remaining: 3m 21s
    724:	learn: 0.2498290	total: 8m 49s	remaining: 3m 20s
    725:	learn: 0.2498269	total: 8m 50s	remaining: 3m 20s
    726:	learn: 0.2498233	total: 8m 50s	remaining: 3m 19s
    727:	learn: 0.2498211	total: 8m 51s	remaining: 3m 18s
    728:	learn: 0.2498191	total: 8m 52s	remaining: 3m 17s
    729:	learn: 0.2498165	total: 8m 52s	remaining: 3m 17s
    730:	learn: 0.2498126	total: 8m 53s	remaining: 3m 16s
    731:	learn: 0.2498105	total: 8m 54s	remaining: 3m 15s
    732:	learn: 0.2498087	total: 8m 54s	remaining: 3m 14s
    733:	learn: 0.2498053	total: 8m 55s	remaining: 3m 14s
    734:	learn: 0.2498019	total: 8m 56s	remaining: 3m 13s
    735:	learn: 0.2498001	total: 8m 57s	remaining: 3m 12s
    736:	learn: 0.2497976	total: 8m 57s	remaining: 3m 11s
    737:	learn: 0.2497962	total: 8m 58s	remaining: 3m 11s
    738:	learn: 0.2497940	total: 8m 59s	remaining: 3m 10s
    739:	learn: 0.2497926	total: 8m 59s	remaining: 3m 9s
    740:	learn: 0.2497901	total: 9m	remaining: 3m 9s
    741:	learn: 0.2497873	total: 9m 1s	remaining: 3m 8s
    742:	learn: 0.2497841	total: 9m 2s	remaining: 3m 7s
    743:	learn: 0.2497804	total: 9m 3s	remaining: 3m 6s
    744:	learn: 0.2497776	total: 9m 3s	remaining: 3m 6s
    745:	learn: 0.2497751	total: 9m 4s	remaining: 3m 5s
    746:	learn: 0.2497738	total: 9m 5s	remaining: 3m 4s
    747:	learn: 0.2497718	total: 9m 6s	remaining: 3m 4s
    748:	learn: 0.2497701	total: 9m 6s	remaining: 3m 3s
    749:	learn: 0.2497678	total: 9m 7s	remaining: 3m 2s
    750:	learn: 0.2497658	total: 9m 8s	remaining: 3m 1s
    751:	learn: 0.2497636	total: 9m 9s	remaining: 3m 1s
    752:	learn: 0.2497614	total: 9m 9s	remaining: 3m
    753:	learn: 0.2497566	total: 9m 10s	remaining: 2m 59s
    754:	learn: 0.2497547	total: 9m 11s	remaining: 2m 58s
    755:	learn: 0.2497527	total: 9m 12s	remaining: 2m 58s
    756:	learn: 0.2497502	total: 9m 12s	remaining: 2m 57s
    757:	learn: 0.2497486	total: 9m 13s	remaining: 2m 56s
    758:	learn: 0.2497467	total: 9m 14s	remaining: 2m 56s
    759:	learn: 0.2497444	total: 9m 15s	remaining: 2m 55s
    760:	learn: 0.2497421	total: 9m 15s	remaining: 2m 54s
    761:	learn: 0.2497394	total: 9m 16s	remaining: 2m 53s
    762:	learn: 0.2497372	total: 9m 17s	remaining: 2m 53s
    763:	learn: 0.2497346	total: 9m 18s	remaining: 2m 52s
    764:	learn: 0.2497318	total: 9m 18s	remaining: 2m 51s
    765:	learn: 0.2497284	total: 9m 19s	remaining: 2m 50s
    766:	learn: 0.2497270	total: 9m 20s	remaining: 2m 50s
    767:	learn: 0.2497251	total: 9m 21s	remaining: 2m 49s
    768:	learn: 0.2497238	total: 9m 21s	remaining: 2m 48s
    769:	learn: 0.2497212	total: 9m 22s	remaining: 2m 48s
    770:	learn: 0.2497193	total: 9m 23s	remaining: 2m 47s
    771:	learn: 0.2497171	total: 9m 24s	remaining: 2m 46s
    772:	learn: 0.2497143	total: 9m 24s	remaining: 2m 45s
    773:	learn: 0.2497118	total: 9m 25s	remaining: 2m 45s
    774:	learn: 0.2497099	total: 9m 26s	remaining: 2m 44s
    775:	learn: 0.2497069	total: 9m 27s	remaining: 2m 43s
    776:	learn: 0.2497049	total: 9m 27s	remaining: 2m 43s
    777:	learn: 0.2497031	total: 9m 28s	remaining: 2m 42s
    778:	learn: 0.2497008	total: 9m 29s	remaining: 2m 41s
    779:	learn: 0.2496990	total: 9m 30s	remaining: 2m 40s
    780:	learn: 0.2496966	total: 9m 30s	remaining: 2m 40s
    781:	learn: 0.2496951	total: 9m 31s	remaining: 2m 39s
    782:	learn: 0.2496930	total: 9m 32s	remaining: 2m 38s
    783:	learn: 0.2496915	total: 9m 33s	remaining: 2m 37s
    784:	learn: 0.2496886	total: 9m 33s	remaining: 2m 37s
    785:	learn: 0.2496864	total: 9m 34s	remaining: 2m 36s
    786:	learn: 0.2496844	total: 9m 35s	remaining: 2m 35s
    787:	learn: 0.2496819	total: 9m 35s	remaining: 2m 34s
    788:	learn: 0.2496796	total: 9m 36s	remaining: 2m 34s
    789:	learn: 0.2496771	total: 9m 37s	remaining: 2m 33s
    790:	learn: 0.2496753	total: 9m 38s	remaining: 2m 32s
    791:	learn: 0.2496731	total: 9m 38s	remaining: 2m 32s
    792:	learn: 0.2496709	total: 9m 39s	remaining: 2m 31s
    793:	learn: 0.2496695	total: 9m 40s	remaining: 2m 30s
    794:	learn: 0.2496681	total: 9m 41s	remaining: 2m 29s
    795:	learn: 0.2496657	total: 9m 41s	remaining: 2m 29s
    796:	learn: 0.2496638	total: 9m 42s	remaining: 2m 28s
    797:	learn: 0.2496622	total: 9m 43s	remaining: 2m 27s
    798:	learn: 0.2496606	total: 9m 43s	remaining: 2m 26s
    799:	learn: 0.2496575	total: 9m 44s	remaining: 2m 26s
    800:	learn: 0.2496550	total: 9m 45s	remaining: 2m 25s
    801:	learn: 0.2496517	total: 9m 46s	remaining: 2m 24s
    802:	learn: 0.2496484	total: 9m 47s	remaining: 2m 24s
    803:	learn: 0.2496455	total: 9m 47s	remaining: 2m 23s
    804:	learn: 0.2496432	total: 9m 48s	remaining: 2m 22s
    805:	learn: 0.2496410	total: 9m 49s	remaining: 2m 21s
    806:	learn: 0.2496386	total: 9m 49s	remaining: 2m 21s
    807:	learn: 0.2496353	total: 9m 50s	remaining: 2m 20s
    808:	learn: 0.2496333	total: 9m 51s	remaining: 2m 19s
    809:	learn: 0.2496299	total: 9m 52s	remaining: 2m 18s
    810:	learn: 0.2496272	total: 9m 52s	remaining: 2m 18s
    811:	learn: 0.2496256	total: 9m 53s	remaining: 2m 17s
    812:	learn: 0.2496234	total: 9m 54s	remaining: 2m 16s
    813:	learn: 0.2496207	total: 9m 55s	remaining: 2m 15s
    814:	learn: 0.2496187	total: 9m 55s	remaining: 2m 15s
    815:	learn: 0.2496163	total: 9m 56s	remaining: 2m 14s
    816:	learn: 0.2496144	total: 9m 57s	remaining: 2m 13s
    817:	learn: 0.2496123	total: 9m 57s	remaining: 2m 13s
    818:	learn: 0.2496106	total: 9m 58s	remaining: 2m 12s
    819:	learn: 0.2496086	total: 9m 59s	remaining: 2m 11s
    820:	learn: 0.2496064	total: 10m	remaining: 2m 10s
    821:	learn: 0.2496047	total: 10m 1s	remaining: 2m 10s
    822:	learn: 0.2496029	total: 10m 1s	remaining: 2m 9s
    823:	learn: 0.2496008	total: 10m 2s	remaining: 2m 8s
    824:	learn: 0.2495995	total: 10m 3s	remaining: 2m 7s
    825:	learn: 0.2495975	total: 10m 3s	remaining: 2m 7s
    826:	learn: 0.2495956	total: 10m 4s	remaining: 2m 6s
    827:	learn: 0.2495936	total: 10m 5s	remaining: 2m 5s
    828:	learn: 0.2495902	total: 10m 6s	remaining: 2m 5s
    829:	learn: 0.2495878	total: 10m 6s	remaining: 2m 4s
    830:	learn: 0.2495864	total: 10m 7s	remaining: 2m 3s
    831:	learn: 0.2495842	total: 10m 8s	remaining: 2m 2s
    832:	learn: 0.2495828	total: 10m 8s	remaining: 2m 2s
    833:	learn: 0.2495806	total: 10m 9s	remaining: 2m 1s
    834:	learn: 0.2495789	total: 10m 10s	remaining: 2m
    835:	learn: 0.2495772	total: 10m 11s	remaining: 1m 59s
    836:	learn: 0.2495757	total: 10m 11s	remaining: 1m 59s
    837:	learn: 0.2495737	total: 10m 12s	remaining: 1m 58s
    838:	learn: 0.2495716	total: 10m 13s	remaining: 1m 57s
    839:	learn: 0.2495691	total: 10m 13s	remaining: 1m 56s
    840:	learn: 0.2495671	total: 10m 14s	remaining: 1m 56s
    841:	learn: 0.2495640	total: 10m 15s	remaining: 1m 55s
    842:	learn: 0.2495621	total: 10m 16s	remaining: 1m 54s
    843:	learn: 0.2495597	total: 10m 17s	remaining: 1m 54s
    844:	learn: 0.2495571	total: 10m 17s	remaining: 1m 53s
    845:	learn: 0.2495557	total: 10m 18s	remaining: 1m 52s
    846:	learn: 0.2495538	total: 10m 19s	remaining: 1m 51s
    847:	learn: 0.2495512	total: 10m 20s	remaining: 1m 51s
    848:	learn: 0.2495489	total: 10m 20s	remaining: 1m 50s
    849:	learn: 0.2495472	total: 10m 21s	remaining: 1m 49s
    850:	learn: 0.2495447	total: 10m 22s	remaining: 1m 48s
    851:	learn: 0.2495424	total: 10m 23s	remaining: 1m 48s
    852:	learn: 0.2495407	total: 10m 23s	remaining: 1m 47s
    853:	learn: 0.2495387	total: 10m 24s	remaining: 1m 46s
    854:	learn: 0.2495360	total: 10m 25s	remaining: 1m 46s
    855:	learn: 0.2495345	total: 10m 25s	remaining: 1m 45s
    856:	learn: 0.2495314	total: 10m 26s	remaining: 1m 44s
    857:	learn: 0.2495284	total: 10m 27s	remaining: 1m 43s
    858:	learn: 0.2495262	total: 10m 27s	remaining: 1m 43s
    859:	learn: 0.2495239	total: 10m 28s	remaining: 1m 42s
    860:	learn: 0.2495218	total: 10m 29s	remaining: 1m 41s
    861:	learn: 0.2495197	total: 10m 30s	remaining: 1m 40s
    862:	learn: 0.2495173	total: 10m 30s	remaining: 1m 40s
    863:	learn: 0.2495147	total: 10m 31s	remaining: 1m 39s
    864:	learn: 0.2495127	total: 10m 32s	remaining: 1m 38s
    865:	learn: 0.2495098	total: 10m 32s	remaining: 1m 37s
    866:	learn: 0.2495071	total: 10m 33s	remaining: 1m 37s
    867:	learn: 0.2495051	total: 10m 34s	remaining: 1m 36s
    868:	learn: 0.2495029	total: 10m 35s	remaining: 1m 35s
    869:	learn: 0.2495005	total: 10m 35s	remaining: 1m 34s
    870:	learn: 0.2494980	total: 10m 36s	remaining: 1m 34s
    871:	learn: 0.2494965	total: 10m 37s	remaining: 1m 33s
    872:	learn: 0.2494946	total: 10m 37s	remaining: 1m 32s
    873:	learn: 0.2494924	total: 10m 38s	remaining: 1m 32s
    874:	learn: 0.2494904	total: 10m 39s	remaining: 1m 31s
    875:	learn: 0.2494878	total: 10m 40s	remaining: 1m 30s
    876:	learn: 0.2494855	total: 10m 41s	remaining: 1m 29s
    877:	learn: 0.2494830	total: 10m 41s	remaining: 1m 29s
    878:	learn: 0.2494810	total: 10m 42s	remaining: 1m 28s
    879:	learn: 0.2494792	total: 10m 43s	remaining: 1m 27s
    880:	learn: 0.2494769	total: 10m 43s	remaining: 1m 26s
    881:	learn: 0.2494738	total: 10m 44s	remaining: 1m 26s
    882:	learn: 0.2494696	total: 10m 45s	remaining: 1m 25s
    883:	learn: 0.2494674	total: 10m 45s	remaining: 1m 24s
    884:	learn: 0.2494652	total: 10m 46s	remaining: 1m 24s
    885:	learn: 0.2494631	total: 10m 47s	remaining: 1m 23s
    886:	learn: 0.2494607	total: 10m 48s	remaining: 1m 22s
    887:	learn: 0.2494576	total: 10m 49s	remaining: 1m 21s
    888:	learn: 0.2494560	total: 10m 49s	remaining: 1m 21s
    889:	learn: 0.2494532	total: 10m 50s	remaining: 1m 20s
    890:	learn: 0.2494510	total: 10m 51s	remaining: 1m 19s
    891:	learn: 0.2494487	total: 10m 51s	remaining: 1m 18s
    892:	learn: 0.2494465	total: 10m 52s	remaining: 1m 18s
    893:	learn: 0.2494441	total: 10m 53s	remaining: 1m 17s
    894:	learn: 0.2494415	total: 10m 54s	remaining: 1m 16s
    895:	learn: 0.2494390	total: 10m 55s	remaining: 1m 16s
    896:	learn: 0.2494373	total: 10m 56s	remaining: 1m 15s
    897:	learn: 0.2494357	total: 10m 56s	remaining: 1m 14s
    898:	learn: 0.2494338	total: 10m 57s	remaining: 1m 13s
    899:	learn: 0.2494316	total: 10m 58s	remaining: 1m 13s
    900:	learn: 0.2494292	total: 10m 59s	remaining: 1m 12s
    901:	learn: 0.2494272	total: 10m 59s	remaining: 1m 11s
    902:	learn: 0.2494238	total: 11m	remaining: 1m 10s
    903:	learn: 0.2494219	total: 11m 1s	remaining: 1m 10s
    904:	learn: 0.2494205	total: 11m 2s	remaining: 1m 9s
    905:	learn: 0.2494170	total: 11m 2s	remaining: 1m 8s
    906:	learn: 0.2494149	total: 11m 3s	remaining: 1m 8s
    907:	learn: 0.2494131	total: 11m 4s	remaining: 1m 7s
    908:	learn: 0.2494078	total: 11m 5s	remaining: 1m 6s
    909:	learn: 0.2494059	total: 11m 5s	remaining: 1m 5s
    910:	learn: 0.2494037	total: 11m 6s	remaining: 1m 5s
    911:	learn: 0.2494015	total: 11m 7s	remaining: 1m 4s
    912:	learn: 0.2493989	total: 11m 8s	remaining: 1m 3s
    913:	learn: 0.2493976	total: 11m 8s	remaining: 1m 2s
    914:	learn: 0.2493961	total: 11m 9s	remaining: 1m 2s
    915:	learn: 0.2493942	total: 11m 10s	remaining: 1m 1s
    916:	learn: 0.2493923	total: 11m 11s	remaining: 1m
    917:	learn: 0.2493901	total: 11m 11s	remaining: 1m
    918:	learn: 0.2493876	total: 11m 12s	remaining: 59.3s
    919:	learn: 0.2493843	total: 11m 13s	remaining: 58.6s
    920:	learn: 0.2493825	total: 11m 13s	remaining: 57.8s
    921:	learn: 0.2493809	total: 11m 14s	remaining: 57.1s
    922:	learn: 0.2493790	total: 11m 15s	remaining: 56.4s
    923:	learn: 0.2493766	total: 11m 16s	remaining: 55.6s
    924:	learn: 0.2493746	total: 11m 17s	remaining: 54.9s
    925:	learn: 0.2493724	total: 11m 17s	remaining: 54.2s
    926:	learn: 0.2493704	total: 11m 18s	remaining: 53.4s
    927:	learn: 0.2493688	total: 11m 19s	remaining: 52.7s
    928:	learn: 0.2493667	total: 11m 20s	remaining: 52s
    929:	learn: 0.2493645	total: 11m 20s	remaining: 51.3s
    930:	learn: 0.2493627	total: 11m 21s	remaining: 50.5s
    931:	learn: 0.2493606	total: 11m 22s	remaining: 49.8s
    932:	learn: 0.2493578	total: 11m 23s	remaining: 49s
    933:	learn: 0.2493555	total: 11m 23s	remaining: 48.3s
    934:	learn: 0.2493525	total: 11m 24s	remaining: 47.6s
    935:	learn: 0.2493506	total: 11m 25s	remaining: 46.9s
    936:	learn: 0.2493483	total: 11m 26s	remaining: 46.1s
    937:	learn: 0.2493461	total: 11m 26s	remaining: 45.4s
    938:	learn: 0.2493446	total: 11m 27s	remaining: 44.7s
    939:	learn: 0.2493427	total: 11m 28s	remaining: 43.9s
    940:	learn: 0.2493401	total: 11m 29s	remaining: 43.2s
    941:	learn: 0.2493381	total: 11m 29s	remaining: 42.5s
    942:	learn: 0.2493361	total: 11m 30s	remaining: 41.7s
    943:	learn: 0.2493332	total: 11m 31s	remaining: 41s
    944:	learn: 0.2493313	total: 11m 32s	remaining: 40.3s
    945:	learn: 0.2493294	total: 11m 32s	remaining: 39.5s
    946:	learn: 0.2493270	total: 11m 33s	remaining: 38.8s
    947:	learn: 0.2493256	total: 11m 34s	remaining: 38.1s
    948:	learn: 0.2493225	total: 11m 35s	remaining: 37.4s
    949:	learn: 0.2493203	total: 11m 35s	remaining: 36.6s
    950:	learn: 0.2493186	total: 11m 36s	remaining: 35.9s
    951:	learn: 0.2493166	total: 11m 37s	remaining: 35.2s
    952:	learn: 0.2493146	total: 11m 37s	remaining: 34.4s
    953:	learn: 0.2493126	total: 11m 38s	remaining: 33.7s
    954:	learn: 0.2493112	total: 11m 39s	remaining: 33s
    955:	learn: 0.2493099	total: 11m 40s	remaining: 32.2s
    956:	learn: 0.2493074	total: 11m 40s	remaining: 31.5s
    957:	learn: 0.2493059	total: 11m 41s	remaining: 30.8s
    958:	learn: 0.2493034	total: 11m 42s	remaining: 30s
    959:	learn: 0.2493002	total: 11m 43s	remaining: 29.3s
    960:	learn: 0.2492982	total: 11m 43s	remaining: 28.6s
    961:	learn: 0.2492962	total: 11m 44s	remaining: 27.8s
    962:	learn: 0.2492943	total: 11m 45s	remaining: 27.1s
    963:	learn: 0.2492917	total: 11m 46s	remaining: 26.4s
    964:	learn: 0.2492898	total: 11m 46s	remaining: 25.6s
    965:	learn: 0.2492877	total: 11m 47s	remaining: 24.9s
    966:	learn: 0.2492853	total: 11m 48s	remaining: 24.2s
    967:	learn: 0.2492834	total: 11m 49s	remaining: 23.4s
    968:	learn: 0.2492812	total: 11m 49s	remaining: 22.7s
    969:	learn: 0.2492791	total: 11m 50s	remaining: 22s
    970:	learn: 0.2492775	total: 11m 51s	remaining: 21.2s
    971:	learn: 0.2492749	total: 11m 51s	remaining: 20.5s
    972:	learn: 0.2492738	total: 11m 52s	remaining: 19.8s
    973:	learn: 0.2492717	total: 11m 53s	remaining: 19s
    974:	learn: 0.2492696	total: 11m 53s	remaining: 18.3s
    975:	learn: 0.2492672	total: 11m 54s	remaining: 17.6s
    976:	learn: 0.2492643	total: 11m 55s	remaining: 16.8s
    977:	learn: 0.2492624	total: 11m 56s	remaining: 16.1s
    978:	learn: 0.2492607	total: 11m 56s	remaining: 15.4s
    979:	learn: 0.2492587	total: 11m 57s	remaining: 14.6s
    980:	learn: 0.2492567	total: 11m 58s	remaining: 13.9s
    981:	learn: 0.2492545	total: 11m 59s	remaining: 13.2s
    982:	learn: 0.2492528	total: 11m 59s	remaining: 12.4s
    983:	learn: 0.2492509	total: 12m	remaining: 11.7s
    984:	learn: 0.2492489	total: 12m 1s	remaining: 11s
    985:	learn: 0.2492466	total: 12m 2s	remaining: 10.3s
    986:	learn: 0.2492435	total: 12m 3s	remaining: 9.52s
    987:	learn: 0.2492417	total: 12m 3s	remaining: 8.79s
    988:	learn: 0.2492404	total: 12m 4s	remaining: 8.06s
    989:	learn: 0.2492377	total: 12m 5s	remaining: 7.33s
    990:	learn: 0.2492357	total: 12m 6s	remaining: 6.59s
    991:	learn: 0.2492345	total: 12m 6s	remaining: 5.86s
    992:	learn: 0.2492323	total: 12m 7s	remaining: 5.13s
    993:	learn: 0.2492290	total: 12m 8s	remaining: 4.39s
    994:	learn: 0.2492273	total: 12m 8s	remaining: 3.66s
    995:	learn: 0.2492251	total: 12m 9s	remaining: 2.93s
    996:	learn: 0.2492223	total: 12m 10s	remaining: 2.2s
    997:	learn: 0.2492203	total: 12m 11s	remaining: 1.47s
    998:	learn: 0.2492177	total: 12m 11s	remaining: 733ms
    999:	learn: 0.2492157	total: 12m 12s	remaining: 0us
    




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;prep&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;scalar&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Age&#x27;, &#x27;Region_Code&#x27;,
                                                   &#x27;Annual_Premium&#x27;,
                                                   &#x27;Policy_Sales_Channel&#x27;,
                                                   &#x27;Vintage&#x27;]),
                                                 (&#x27;categoric&#x27;,
                                                  Pipeline(steps=[(&#x27;onehot&#x27;,
                                                                   OneHotEncoder())]),
                                                  [&#x27;Gender&#x27;, &#x27;Driving_License&#x27;,
                                                   &#x27;Previously_Insured&#x27;,
                                                   &#x27;Vehicle_Age&#x27;,
                                                   &#x27;Vehicle_Damage&#x27;])])),
                (&#x27;algo&#x27;,
                 &lt;catboost.core.CatBoostClassifier object at 0x000001A5BC1FDC90&gt;)])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;prep&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;scalar&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Age&#x27;, &#x27;Region_Code&#x27;,
                                                   &#x27;Annual_Premium&#x27;,
                                                   &#x27;Policy_Sales_Channel&#x27;,
                                                   &#x27;Vintage&#x27;]),
                                                 (&#x27;categoric&#x27;,
                                                  Pipeline(steps=[(&#x27;onehot&#x27;,
                                                                   OneHotEncoder())]),
                                                  [&#x27;Gender&#x27;, &#x27;Driving_License&#x27;,
                                                   &#x27;Previously_Insured&#x27;,
                                                   &#x27;Vehicle_Age&#x27;,
                                                   &#x27;Vehicle_Damage&#x27;])])),
                (&#x27;algo&#x27;,
                 &lt;catboost.core.CatBoostClassifier object at 0x000001A5BC1FDC90&gt;)])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;prep: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for prep: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                 Pipeline(steps=[(&#x27;scalar&#x27;, StandardScaler())]),
                                 [&#x27;Age&#x27;, &#x27;Region_Code&#x27;, &#x27;Annual_Premium&#x27;,
                                  &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;]),
                                (&#x27;categoric&#x27;,
                                 Pipeline(steps=[(&#x27;onehot&#x27;, OneHotEncoder())]),
                                 [&#x27;Gender&#x27;, &#x27;Driving_License&#x27;,
                                  &#x27;Previously_Insured&#x27;, &#x27;Vehicle_Age&#x27;,
                                  &#x27;Vehicle_Damage&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric</label><div class="sk-toggleable__content fitted"><pre>[&#x27;Age&#x27;, &#x27;Region_Code&#x27;, &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;StandardScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">categoric</label><div class="sk-toggleable__content fitted"><pre>[&#x27;Gender&#x27;, &#x27;Driving_License&#x27;, &#x27;Previously_Insured&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder()</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CatBoostClassifier</label><div class="sk-toggleable__content fitted"><pre>&lt;catboost.core.CatBoostClassifier object at 0x000001A5BC1FDC90&gt;</pre></div> </div></div></div></div></div></div>




```python
# Predict probabilities on the test set
X_test_transformed = pipeline_catboost.named_steps['prep'].transform(X_test)
y_test_pred_proba = pipeline_catboost.named_steps['algo'].predict_proba(X_test_transformed)[:, 1]
```

# Evaluation with ROC AUC Score


```python
# Evaluate with ROC AUC Score
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f'ROC AUC Score: {roc_auc:.4f}')
```

    ROC AUC Score: 0.8808
    


```python
# Calculate and Visualize ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```


    
![png](output_15_0.png)
    



```python
# Display feature importance

catboost_model = pipeline_catboost.named_steps['algo']
feature_importances = catboost_model.get_feature_importance()
feature_names = (pipeline_catboost.named_steps['prep'].named_transformers_['categoric'].named_steps['onehot'].get_feature_names_out().tolist() + 
                 pipeline_catboost.named_steps['prep'].named_transformers_['numeric'].feature_names_in_.tolist())

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')

# Add labels to the bars
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')

plt.show()
```


    
![png](output_16_0.png)
    


# Apply to New Dataset


```python
# Predict probabilities on the actual test dataset
X_test_new = pipeline_catboost.named_steps['prep'].transform(df_test)
y_test_new_pred_proba = pipeline_catboost.named_steps['algo'].predict_proba(X_test_new)[:, 1]
```


```python
# Create a DataFrame for the results
df_results = df_test.copy()
df_results['Response'] = y_test_new_pred_proba
df_results = df_results[['Response']].reset_index()
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11504798</td>
      <td>0.005105</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11504799</td>
      <td>0.476252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11504800</td>
      <td>0.203916</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11504801</td>
      <td>0.000065</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11504802</td>
      <td>0.061538</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7669861</th>
      <td>19174659</td>
      <td>0.206845</td>
    </tr>
    <tr>
      <th>7669862</th>
      <td>19174660</td>
      <td>0.000101</td>
    </tr>
    <tr>
      <th>7669863</th>
      <td>19174661</td>
      <td>0.000198</td>
    </tr>
    <tr>
      <th>7669864</th>
      <td>19174662</td>
      <td>0.589880</td>
    </tr>
    <tr>
      <th>7669865</th>
      <td>19174663</td>
      <td>0.000091</td>
    </tr>
  </tbody>
</table>
<p>7669866 rows × 2 columns</p>
</div>




```python
# Save the predictions to a CSV file
df_results.to_csv(r"C:\Users\ACER\Downloads\Binary Classification of Insurance Cross Selling (CatBoost Model).csv", index=False)

print("Predictions saved to 'predictions.csv'.")
```

    Predictions saved to 'predictions.csv'.
    


```python

```
