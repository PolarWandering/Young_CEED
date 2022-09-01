import numpy as np
import pandas as pd
from pmagpy import pmag, ipmag

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
from shapely.geometry import Polygon

from vgptools.auxiliar import spherical2cartesian, shape, eigen_decomposition


def running_mean_APWP (data, plon_label, plat_label, age_label, window_length, time_step, max_age, min_age):
    """
    function to generate running mean APWP..
    """
    
    mean_pole_ages = np.arange(min_age, max_age + time_step, time_step)
    
    running_means = pd.DataFrame(columns=['age','N','n_studies','k','A95','csd','plon','plat'])
    
    for age in mean_pole_ages:
        
        window_min = age - (window_length / 2.)
        window_max = age + (window_length / 2.)
            
        poles = data.loc[(data[age_label] >= window_min) & (data[age_label] <= window_max)]
        
        number_studies = len(poles['Study'].unique())
        mean = ipmag.fisher_mean(dec=poles[plon_label].tolist(), inc=poles[plat_label].tolist())

        if mean: # this just ensures that dict isn't empty
            running_means.loc[age] = [age, mean['n'], number_studies, mean['k'],mean['alpha95'], mean['csd'], mean['dec'], mean['inc']]
    
    running_means.reset_index(drop=1, inplace=True)
    
    return running_means


def RM_stats(df):
      
    fig, ax = plt.subplots(figsize=(15,3))
    ax2 = ax.twinx()
    # plt.title('Age distribution of Paleomagnetic Poles')
    # plt.ylabel('Number of Paleomagentic Poles')
    plt.xlabel('Mean Age')

    df['kappa_norm'] = df['k'] / df['k'].max()
    df['N_norm'] = df['N'] / df['N'].max()

    dfm = df[['age', 'A95', 'n_studies', 'csd']].melt('age', var_name='type', value_name='vals')


    sns.lineplot(data  = dfm, x = dfm['age'], y = dfm['vals'], hue = dfm['type'], marker="o", ax=ax)
    
    sns.lineplot(data  = df, x = df['age'], y = df['k'], marker="o",  ax=ax2, color= "r")
    
    # ax2.legend(handles=[a.lines[0] for a in [ax,ax2]], 
    #        labels=["kappa"])
    plt.show()
    
    
    
def running_mean_APWP_shape(data, plon_label, plat_label, age_label, window_length, time_step, max_age, min_age):
    """
    function to generate running mean APWP..
    """
    
    mean_pole_ages = np.arange(min_age, max_age + time_step, time_step)
    
    running_means = pd.DataFrame(columns=['age','N','n_studies','k','A95','csd','plon','plat', 'foliation','lineation','collinearity','coplanarity'])
    
    for age in mean_pole_ages:
        window_min = age - (window_length / 2.)
        window_max = age + (window_length / 2.)
        poles = data.loc[(data[age_label] >= window_min) & (data[age_label] <= window_max)]
        number_studies = len(poles['Study'].unique())
        mean = ipmag.fisher_mean(dec=poles[plon_label].tolist(), inc=poles[plat_label].tolist())
        
        ArrayXYZ = np.array([spherical2cartesian([i[plat_label], i[plon_label]]) for _,i in poles.iterrows()])        
        if len(ArrayXYZ) > 3:
            shapes = shape(ArrayXYZ)       
        else:
            shapes = [np.nan,np.nan,np.nan,np.nan]
        
        if mean: # this just ensures that dict isn't empty
            running_means.loc[age] = [age, mean['n'], number_studies, mean['k'],mean['alpha95'], mean['csd'], mean['dec'], mean['inc'], 
                                      shapes[0], shapes[1], shapes[2], shapes[3]]
    
    running_means.reset_index(drop=1, inplace=True)
    
    return running_means