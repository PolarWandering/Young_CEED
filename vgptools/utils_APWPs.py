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

def get_pseudo_vgps(df):  
    '''
    takes a DF with paleomagnetic poles and respective statistics, it draws N randomly generated VGPs
    following the pole location and kappa concentration parameter. In the present formulation we follow
    a very conservative apporach for the assignaiton of ages to each VGP, it is taken at random between
    the lower and upper bounds of the distribution of reported VGPs.
    Note: column labels are presently hard-coded into this, if relevant.
    '''
    Study, age_bst, vgp_lat_bst, vgp_lon_bst = [], [], [], []

    for index, row in df.iterrows():
        
        # we first generate N VGPs following with N the number of VGPs from the original pole.
        directions_temp = ipmag.fishrot(k = row.K, n = row.N, dec = row.Plon, inc = row.Plat, di_block = False)
        
        vgp_lon_bst.append(directions_temp[0])
        vgp_lat_bst.append(directions_temp[1])
    
        age_bst.append([np.random.randint(np.floor(row.min_age),np.ceil(row.max_age)) for _ in range(row.N)])
        Study.append([row.Study for _ in range(row.N)])
    
    vgp_lon_bst = [item for sublist in vgp_lon_bst for item in sublist]
    vgp_lat_bst = [item for sublist in vgp_lat_bst for item in sublist] 
    age_bst = [item for sublist in age_bst for item in sublist]
    Study = [item for sublist in Study for item in sublist]
    
    dictionary = {
                  'Study': Study,
                  'Plat': vgp_lat_bst,    
                  'Plon': vgp_lon_bst,
                  'mean_age': age_bst
                  }    
    
    pseudo_vgps = pd.DataFrame(dictionary)

    return pseudo_vgps

def get_vgps_sampling_direction(df):
    '''
    takes a DF with site information, it draws for each direction a random direction following the
    kappa concentration parameter and mean direction. Then, it calculates from the random direction a given VGP
    In the present formulation we follow a conservative approach for the assignaiton of ages to each direction/VGP, 
    it is taken at random between the min_age and max_age of reported VGPs.
    '''    
    Study, age_bst, decs, incs, slat, slon = [], [], [], [], [], []
    k_mean = df['k'].mean()
    
    for index, row in df.iterrows():        
        # we first generate one random direction from the original entry.
        kappa = k_mean if np.isnan(row.k) else row.k # if we don't have kappa, we take the mean of the reported ones       
        directions_temp = ipmag.fishrot(k = kappa, n = 1, dec = row.dec, inc = row.inc, di_block = False)
        
        decs.append(directions_temp[0][0])
        incs.append(directions_temp[1][0])
        slat.append(row.slat)
        slon.append(row.slon)
        
        age_bst.append(np.random.randint(np.floor(row.min_age),np.ceil(row.max_age)))
        Study.append(row.Study)
        
    dictionary = {
                  'Study': Study,
                  'age': age_bst,
                  'dec': decs,    
                  'inc': incs,
                  'slat': slat,
                  'slon': slon 
                  }    
    new_df = pd.DataFrame(dictionary)        
    new_df['plon'] = new_df.apply(lambda row: pmag.dia_vgp(row.dec, row.inc, 1, row.slat, row.slon)[0], axis =1)
    new_df['plat'] = new_df.apply(lambda row: pmag.dia_vgp(row.dec, row.inc, 1, row.slat, row.slon)[1], axis =1)
    # transform to the southern hemisphere
    new_df['plat'] = np.where(new_df['plat'] > 0, -new_df['plat'], new_df['plat'])
    new_df['plon'] = np.where(new_df['plon'] > 0,(new_df['plon'] - 180.) % 360., new_df['plon'])             

    return new_df

def running_mean_VGPs_bootstrapped(df_vgps, plon_label, plat_label, age_label, window_length, time_step, max_age, min_age, n_bst = 100):
    '''
    takes a compilation of vgps and for each time window uses the bootstrap approach to construct empirical confidence bounds. 
    '''

    running_means_global = pd.DataFrame(columns=['run','N','k','A95','csd','foliation','lineation','collinearity','coplanarity'])

    for i in range(n_bst):
               
        vgps_sample = df_vgps.sample(n = len(df_vgps), replace = True)
        running_means_tmp = pd.DataFrame()
        running_means_tmp = running_mean_APWP_shape(vgps_sample, 'vgp_lon_SH', 'vgp_lat_SH', 'mean_age', window_length, time_step, max_age, min_age)
        running_means_tmp['run'] = float(i)
        running_means_global = running_means_global.append(running_means_tmp, ignore_index=True)
       
    return running_means_global