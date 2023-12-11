import grand.dataio.root_trees as rt
import uproot
import numpy as np
import datetime
import matplotlib.dates as mdates
import math
import ROOT
import os
import ast
import scipy
from datetime import timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from webpages_defs_parallel import *


day_week_month_list = ['1_day', '7_days', '30_days']#'1_day', '7_days', '30_days'
GA_or_GP13_list = ['GP13','GA'] # 'GA','GP13'

# Get the current date in "YYYY/MM/DD" format
current_date = datetime.datetime.now().strftime('%Y/%m/%d')
dir_str = f'/pbs/home/a/atimmerm/GRAND/data/YMD_GRANDfiles/{current_date}/'#f'/sps/grand/data/auger/YMD_GRANDfiles/{current_date}/'




for GA_or_GP13 in GA_or_GP13_list:
    for day_week_month in day_week_month_list:
        bat_temp_html(dir_str, day_week_month, GA_or_GP13)
        avg_freq_trace_HTML(dir_str, day_week_month, GA_or_GP13)
        RMS_HTML(dir_str, day_week_month, GA_or_GP13)


# # active_GA_antennas_html(dir_str)
active_GP13_antennas_html(dir_str)
# # active_percentage_GA_antennas_html(dir_str)
active_percentage_GP13_antennas_html(dir_str)

for GA_or_GP13 in GA_or_GP13_list:
    day_week_month = '1_day'
    avg_freq_trace_HTML_day_night(dir_str, day_week_month, GA_or_GP13)




# # Get today's date
# today = datetime.datetime.now().date()

# # Generate a list of the last 30 days in 'YYYY_MM_DD' format
# dates = [(today - timedelta(days=i+3)).strftime('%Y/%m/%d') for i in range(7)]


# # Create the dir_str with the current date
# dir_strs = [f'/sps/grand/data/auger/YMD_GRANDfiles/{date}/' for date in dates]


# for dir_str in dir_strs:
#     for GA_or_GP13 in GA_or_GP13_list:
#         for day_week_month in day_week_month_list:
#         #     active_GA_antennas_html(dir_str,day_week_month)
#         #     active_GP13_antennas_html(dir_str,day_week_month)
#         #     active_percentage_GA_antennas_html(dir_str, day_week_month)
#             # active_percentage_GP13_antennas_html(dir_str, day_week_month)
#             # bat_temp_html(dir_str, day_week_month, GA_or_GP13)
#             avg_freq_trace_HTML(dir_str, day_week_month, GA_or_GP13)

# antennas_GA_txt(dir_str)

# antennas_GA_html(dir_str)
# antennas_GP13_html(dir_str)

