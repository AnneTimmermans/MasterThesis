#!/usr/bin/env python
# coding: utf-8

# In[1]:


import grand.dataio.root_trees as rt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os.path
import glob
import numpy as np
from scipy.stats import norm
import ROOT
import datetime
import pandas as pd

from grand import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
)
from grand import ECEF, Geodetic, GRANDCS, LTP
from grand import Geomagnet

import numpy as np
import datetime
import time

import matplotlib.pyplot as plt
import os



def antenna_locations_geodetic(dir_str, plot = False):
    '''
    Input: 
            dir_str = directory string in which the root files of the events are
            plot = boolean, to state whether or not you want to plot the locations
    Output: 
            3 dictionaries of latitude, longitude and altitude of the antennas
            GPS_LAT = { filename : [du, lat, max difference lat]}
            GPS_LONG = { filename : [du, long, max difference long]}
            GPS_ALT = { filename : [du, alt, max difference alt]}
    '''
    directory = os.fsencode(dir_str)


    GPS_LAT = {}
    GPS_LONG = {}
    GPS_ALT = {}

    if plot:
        fig, ax = plt.subplots()
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".root") and filename.startswith("t"): 
            fn = dir_str + filename #"td002010_f0001.root"#
            tf = ROOT.TFile(fn)
            df = rt.DataFile(fn)

            tadc  = rt.TADC(fn)
            trawv = df.trawvoltage

            count = trawv.draw('du_id',"")
            du_id = np.unique(np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(int))
            du_id = np.trim_zeros(du_id) #remove du 0 as it gives wrong data :(


            GPS_lat = {}
            GPS_long = {}
            GPS_alt = {}
            GPS_time = {}

            mean_lats = []
            mean_longs = []
            mean_alts = []

            for du in du_id:
                count = trawv.draw('gps_lat : gps_long : gps_alt : gps_time',"du_id == {}".format(du))

                gps_lat = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(float)
                gps_long = np.array(np.frombuffer(trawv.get_v2(), count=count)).astype(float)+360 # Grand cs does not accept negative longitudes
                gps_alt = np.array(np.frombuffer(trawv.get_v3(), count=count)).astype(float)
                gps_time = np.array(np.frombuffer(trawv.get_v4(), count=count)).astype(float)

                GPS_lat[du] = [gps_lat]
                GPS_long[du] = [gps_long]
                GPS_alt[du] = [gps_alt]
                GPS_time[du] = [gps_time]

                # To set the origin of the grand coordinate system we determine the mean of the locations
                mean_lats.append([du, np.mean(gps_lat), max(gps_lat)-min(gps_lat)])
                mean_longs.append([du, np.mean(gps_long), max(gps_long)-min(gps_long)])
                mean_alts.append([du, np.mean(gps_alt), max(gps_alt)-min(gps_alt)])

            GPS_LAT[filename] = mean_lats
            GPS_LONG[filename] = mean_longs
            GPS_ALT[filename] = mean_alts
            
            if plot:
                for i in range(len(mean_lats)):
                    du, lat, lat_diff = mean_lats[i]
                    du, long, long_diff = mean_longs[i]
                    
                    
                    sc = ax.scatter(lat, long, s=100, alpha = 0.5, color = 'green')
                    ax.annotate("du {}".format(du) , (lat, long))
    
    if plot:
        plt.title("du's  and their location for different files")
        plt.xlabel("latitude")
        plt.ylabel("longitude")
        plt.show()
                    

    return GPS_LAT, GPS_LONG, GPS_ALT


def antenna_locations_grandcs(dir_str, plot_grandcs = False, plot_geodetic = False):
    
    GRAND_X = {}
    GRAND_Y = {}
    GRAND_Z = {}
    
    if plot_grandcs:
        fig, ax = plt.subplots()
    
    # set grand_origin
    mean_lat = -35.1134566
    mean_long = -69.5253518 + 360
    mean_alt = 1551.826541666667
    grand_origin = Geodetic(latitude=mean_lat, longitude=mean_long, height=mean_alt)
    
    
    # make dictionary of geodetic coordinates
    GPS_LAT, GPS_LONG, GPS_ALT = antenna_locations_geodetic(dir_str, plot = plot_geodetic)
    
    values_lat = list(GPS_LAT.values())
    values_long = list(GPS_LONG.values())
    values_alt = list(GPS_ALT.values())
    DU_LAT = {}
    DU_LONG = {}
    DU_ALT = {}
    for i in range(len(values_lat)):
        for j in range(len(values_lat[i])):
            if values_lat[i][j][0] in DU_LAT.keys():

                old_lat = DU_LAT[values_lat[i][j][0]] 
                old_lat.append(values_lat[i][j][1])
                DU_LAT[values_lat[i][j][0]] = old_lat

                old_long = DU_LONG[values_long[i][j][0]] 
                old_long.append(values_long[i][j][1])
                DU_LONG[values_long[i][j][0]] = old_long
                
                old_alt = DU_ALT[values_alt[i][j][0]] 
                old_alt.append(values_alt[i][j][1])
                DU_ALT[values_alt[i][j][0]] = old_alt

            else:
                DU_LAT[values_lat[i][j][0]] = [values_lat[i][j][1]]
                DU_LONG[values_long[i][j][0]] = [values_long[i][j][1]]
                DU_ALT[values_alt[i][j][0]] = [values_alt[i][j][1]]
                
    for du in DU_LAT.keys():
        latitude = np.array(DU_LAT[du])
        longitude = np.array(DU_LONG[du])
        altitude = np.array(DU_ALT[du]) 

        # Conversion from Geodetic to GRAND coordinate system.
        geod = Geodetic(latitude=latitude, longitude=longitude, height=altitude)
        gcs = GRANDCS(geod, location=grand_origin)

        grand_x = np.array((gcs.x)) #/ 1000.0  # m -> km
        grand_y = np.array((gcs.y)) #/ 1000.0  # m -> km
        grand_z = np.array((gcs.z)) #/ 1000.0  # m -> km
        
        GRAND_X[du] = grand_x
        GRAND_Y[du] = grand_y
        GRAND_Z[du] = grand_z
        
        print("Du: {} \n grand x: {} +- {} m \n grand y: {} +- {} m \n \n".format(du, np.mean(grand_x), np.std(grand_x), np.mean(grand_y), np.std(grand_y)))
        
        if plot_grandcs:
            for i in range(len(grand_x)):
                sc = ax.scatter(grand_x[i], grand_y[i], s=100, alpha = 0.5, color = 'green')
                ax.annotate("du {}".format(du) , (grand_x[i], grand_y[i]))

    if plot_grandcs:
        plt.title("du's  and their location for different files")
        plt.xlabel("North [m]")
        plt.ylabel("West [m]")
        plt.show()
    
    return GRAND_X, GRAND_Y, GRAND_Z











