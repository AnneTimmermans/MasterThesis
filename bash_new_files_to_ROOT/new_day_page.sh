#!/bin/bash

#active antennas
cd /pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/active_antennas

cp active_GA_antennas_5_days.html active_GA_antennas_5_days.html 
cp active_GA_antennas_3_days.html active_GA_antennas_4_days.html 
cp active_GA_antennas_2_days.html active_GA_antennas_3_days.html 
cp active_GA_antennas.html active_GA_antennas_2_days.html 


cp active_GP13_antennas_4_days.html active_GP13_antennas_5_days.html 
cp active_GP13_antennas_3_days.html active_GP13_antennas_4_days.html 
cp active_GP13_antennas_2_days.html active_GP13_antennas_3_days.html 
cp active_GP13_antennas.html active_GP13_antennas_2_days.html

cp active_perc_GA_antennas_4_days.html active_perc_GA_antennas_5_days.html 
cp active_perc_GA_antennas_3_days.html active_perc_GA_antennas_4_days.html 
cp active_perc_GA_antennas_2_days.html active_perc_GA_antennas_3_days.html 
cp active_perc_GA_antennas.html active_perc_GA_antennas_2_days.html 


cp active_perc_GP13_antennas_4_days.html active_perc_GP13_antennas_5_days.html 
cp active_perc_GP13_antennas_3_days.html active_perc_GP13_antennas_4_days.html 
cp active_perc_GP13_antennas_2_days.html active_perc_GP13_antennas_3_days.html 
cp active_perc_GP13_antennas.html active_perc_GP13_antennas_2_days.html

echo "active_antennas webpages moved"

#average traces and frequency spectra
cd /pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/avg_trace_freq

cp avg_trace_freq_GA_4_days.html avg_trace_freq_GA_5_days.html 
cp avg_trace_freq_GA_3_days.html avg_trace_freq_GA_4_days.html 
cp avg_trace_freq_GA_2_days.html avg_trace_freq_GA_3_days.html 
cp avg_trace_freq_GA_1_day.html avg_trace_freq_GA_2_days.html 

cp avg_trace_freq_GP13_4_days.html avg_trace_freq_GP13_5_days.html 
cp avg_trace_freq_GP13_3_days.html avg_trace_freq_GP13_4_days.html 
cp avg_trace_freq_GP13_2_days.html avg_trace_freq_GP13_3_days.html 
cp avg_trace_freq_GP13_1_day.html avg_trace_freq_GP13_2_days.html 

echo "avg_trace_freq webpages moved"

#Battery and temperature levels
cd /pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/bat_temp

cp bat_temp_GA_4_days.html bat_temp_GA_5_days.html 
cp bat_temp_GA_3_days.html bat_temp_GA_4_days.html 
cp bat_temp_GA_2_days.html bat_temp_GA_3_days.html 
cp bat_temp_GA_1_day.html bat_temp_GA_2_days.html 

cp bat_temp_GP13_4_days.html bat_temp_GP13_5_days.html 
cp bat_temp_GP13_3_days.html bat_temp_GP13_4_days.html 
cp bat_temp_GP13_2_days.html bat_temp_GP13_3_days.html 
cp bat_temp_GP13_1_day.html bat_temp_GP13_2_days.html 

echo "bat_temp webpages moved"

#RMS levels 
cd /pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/RMS

cp rms_linear_GA_4_days.html rms_linear_GA_5_days.html 
cp rms_linear_GA_3_days.html rms_linear_GA_4_days.html 
cp rms_linear_GA_2_days.html rms_linear_GA_3_days.html 
cp rms_linear_GA_1_day.html rms_linear_GA_2_days.html 

cp rms_linear_GP13_4_days.html rms_linear_GP13_5_days.html 
cp rms_linear_GP13_3_days.html rms_linear_GP13_4_days.html 
cp rms_linear_GP13_2_days.html rms_linear_GP13_3_days.html 
cp rms_linear_GP13_1_day.html rms_linear_GP13_2_days.html 


echo "RMS webpages moved"
