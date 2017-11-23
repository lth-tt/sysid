#!/usr/bin/env python

import pandas as pd
def read_data(control_output,task_vel):
    df_soll     = pd.read_csv(control_output, header = 0, names = ['time', 'x_soll'])
    df_soll     = df_soll.set_index('time')
    df_soll     = df_soll[~df_soll.index.duplicated(keep = 'first')] #gets rid of any duplicate index values if present
    df_ist      = pd.read_csv(task_vel, header = 0, names = ['time', 'x_ist'])
    df_ist      = df_ist.set_index('time')
    df_ist      = df_ist[~df_ist.index.duplicated(keep = 'first')]
    df_ist_soll = pd.concat([df_soll.x_soll, df_ist.x_ist], axis = 1).fillna(method = 'pad')
    df_ist_soll = df_ist_soll.fillna(0)
    return df_ist_soll