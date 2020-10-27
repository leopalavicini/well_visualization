# Well Visualization - FORCE Dataset

In this project, I developed an interactive dashboard to visualize the Well Log data in the Force Dataset, this in three views: a Log Plot, a Map View and a Cross-plot. This dashboard was done using Plotly and Dash.

Inside this repository, you can find the jupyter notebook, well_visual, where there is an step-by-step guide of what I did to develop the dashboard. And there is also a python executable if you don't want to run it on jupyter, this will be easier as you need some extensions for Jupyter to run Dash. 

## Dataset
FORCE and XEEK released a well log dataset with more than 100 wells for their 2020 Machine Learning contest, with each well contanaing a set of well logs, a facies interpretation and their location.

To download the dataset, you can go [here](https://xeek.ai/challenges/force-well-logs/data). The well log data is licensed by Norwegian License for Open Government Data (NLOD) 2.0. and the facies interpretation done by FORCE is licensed as CC-BY-4.0.

## Requirements 
There is a requirements.txt file where you can find all the packages used to generate the dashboard. But, as I mentioned before, if you want to run it in Jupyter, you will need to install plotly and dash extensions for Jupyter. 
