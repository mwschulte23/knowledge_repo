---
title: A simple task recommender & network graph
authors:
- Mike Schulte
tags:
- recommendation
- sp
created_at: 2016-06-29 00:00:00
updated_at: 2019-07-10 17:24:16.338709
tldr: Simple recommedation engine using item-item approach
---

### Motivation

Wanted to identify similar tasks based on way an SP is profiled, i.e group of SPs share tasks but not all, to aid in upselling tasks.  Idea of this post is to show this relationship visually with a network graph

*This notebook skips the recommendation table creation*


```python
import os
import snowflake.connector as sf

import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('fivethirtyeight')

import plotly.plotly as py
import plotly.graph_objs as go
import plotly as pl
pl.tools.set_credentials_file(username = 'mwschulte23', api_key = 'cyZTgFgDA2bFBCLksCsm')
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

import networkx as nx

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>


### SQL Query


```python
query = '''
select sp.user_id sp_id
     , st.task_oid
     , td.task task_name
     , td.pwc pwc_name
  from landing.wisen_data.ws_sp_profile sp
     , landing.wisen_data.sm_service_task st
     , rpt.reports.task_dim td
 where sp.user_id = st.sp_id
   and sp.service_profile_id = st.service_profile_id
   and st.task_oid = td.task_oid
   and sp.service_profile_id = 1
   and sp.status_code = 3
   and sp.accept_type = 2
'''
```
### Data Load & Quick Look


```python
def data_load(query):
    conn = sf.connect(user = os.getenv('SF_USER'), password = os.getenv('SF_PASSWORD'), account = 'homeadvisor')
    out_df = pd.read_sql(query, conn)
    conn.close()
    
    return out_df

df1 = data_load(query)

print('There are {0:,.0f} rows and {1:,.0f} columns'.format(df1.shape[0], df1.shape[1]))
df1.head()
```
    There are 1,142,087 rows and 4 columns
    





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
      <th>SP_ID</th>
      <th>TASK_OID</th>
      <th>TASK_NAME</th>
      <th>PWC_NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70628652</td>
      <td>46426</td>
      <td>Gas Furnace / Forced Air Heating System - Repair</td>
      <td>HVAC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>64886659</td>
      <td>40015</td>
      <td>Concrete Driveways &amp; Floors - Install</td>
      <td>Concrete &amp; Masonry</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35370454</td>
      <td>46426</td>
      <td>Gas Furnace / Forced Air Heating System - Repair</td>
      <td>HVAC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54694253</td>
      <td>61286</td>
      <td>Pest Control - Termite - For Business</td>
      <td>Pest Control</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69745247</td>
      <td>39763</td>
      <td>Appliance (Major Electric Appliance) - Install...</td>
      <td>Appliances</td>
    </tr>
  </tbody>
</table>
</div>



### Data Prep

For this simple exploration, we want to remove SPs with only a few tasks as well as remove tasks with only a few SPs profiled. Below code filters out a percentile of both SPs & tasks


```python
def top_list(df, group, percentile):
    '''get list of top X percentile of tasks or SPs'''
    cols = ['TASK_OID', 'SP_ID']
    
    if group == 'SP_ID':
        counts = df1.groupby(cols[1]).count()[cols[0]].to_frame()
        cut_off = counts[cols[0]].quantile(percentile)
        top = list(counts.loc[counts[cols[0]] >= cut_off].index)
    elif group == 'TASK_OID':
        counts = df1.groupby(cols[0]).count()[cols[1]].to_frame()
        cut_off = counts[cols[1]].quantile(percentile)
        top = list(counts.loc[counts[cols[1]] >= cut_off].index)
    
    return top

top_tasks = top_list(df1, 'TASK_OID', .5)
top_sps = top_list(df1, 'SP_ID', .25)

top_group = df1.loc[(df1['TASK_OID'].isin(top_tasks)) & (df1['SP_ID'].isin(top_sps))]

wide_sp_task = pd.crosstab(index = top_group['TASK_OID'], columns = top_group['SP_ID'])
wide_sp_task.index = pd.Series(wide_sp_task.index).map(df1.groupby('TASK_OID').last()['TASK_NAME'])
wide_st_sparse = csr_matrix(wide_sp_task)
```
### Creating Recommendations


```python
NEIGHBORS = 4

nn = NearestNeighbors(metric = 'cosine')
nn.fit(wide_st_sparse)

distance, indices = nn.kneighbors(wide_sp_task, n_neighbors = NEIGHBORS)
```

```python
def flattened_table(df, indices, distance):
    reco_list = []

    for i in range(df.shape[0]):
        x = pd.DataFrame({'SIMILAR_TASK':df.index[indices][i][1:],
                          'SIMILARITY':distance[i][1:]}, index = [df.index[indices][i][0]] * (NEIGHBORS - 1))
        reco_list.append(x)

    out_df = pd.concat(reco_list).reset_index().rename(columns = {'index': 'TARGET_TASK'})
    
    out_df = out_df.loc[out_df['SIMILARITY'] > 0]
    
    return out_df

similarity_df = flattened_table(wide_sp_task, indices, distance) 
```

```python
similarity_df.head()
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
      <th>TARGET_TASK</th>
      <th>SIMILAR_TASK</th>
      <th>SIMILARITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Appliance (Major Electric Appliance) - Install...</td>
      <td>Appliance (Microwave Oven) - Install or Replace</td>
      <td>0.324</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Appliance (Major Electric Appliance) - Install...</td>
      <td>Appliance (Smaller Size) - Install or Replace</td>
      <td>0.329</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Appliance (Major Electric Appliance) - Install...</td>
      <td>Appliances (All Types) - Repair or Service</td>
      <td>0.639</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Appliances (All Types) - Repair or Service</td>
      <td>Appliance (Major Electric Appliance) - Install...</td>
      <td>0.639</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Appliances (All Types) - Repair or Service</td>
      <td>Appliance (Smaller Size) - Install or Replace</td>
      <td>0.648</td>
    </tr>
  </tbody>
</table>
</div>



Our model created two arrays:
* indices: functionally the 'location' in a table where most similar tasks live - shape = n * NEIGHBORS
* distance: the cosine distance of each task, this is same shape as indices array

With these arrays, I created a flattened table with task pairs & similiarity measurement (0 is closest, 1 is furthest)

### Creating the Network Graph


```python
grid = pd.DataFrame(nn.kneighbors_graph(wide_st_sparse, n_neighbors = 4).toarray().astype(int),
                    index = wide_sp_task.index, columns = wide_sp_task.index)

keys = [i for i in range(grid.shape[0])]
vals = list(grid.index)

graph_labels = dict(zip(keys, vals))

G = nx.from_numpy_matrix(grid.values)
pos = nx.fruchterman_reingold_layout(G)

xn = []
yn = []

for i in pos:
    xn.append(pos[i][0])
    yn.append(pos[i][1])

node_trace = go.Scatter(x = xn, 
                        y = yn, 
                        mode = 'markers',
                        text = list(graph_labels.values()),
                        hoverinfo = 'text',
                        marker = dict(color = 'rgba(17, 157, 255, 0.5)',
                                      size = 8))

x = []
y = []

for i, j in G.edges():
    x0, y0 = pos[i][0], pos[j][0]
    x1, y1 = pos[i][1], pos[j][1]
    x += [x0, y0, None]
    y += [x1, y1, None]
    
edge_trace = go.Scatter(x= x,
                        y= y,
                        line=dict(width = .5, color = '#888'),
                        hoverinfo = 'none',
                        mode = 'lines')

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Task Recommendation - Network Graph',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))


py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~mwschulte23/262.embed" height="525px" width="100%"></iframe>