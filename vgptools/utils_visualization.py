import numpy as np
import pandas as pd
from pmagpy import pmag, ipmag

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
from shapely.geometry import Polygon
from vgptools.auxiliar import eigen_decomposition, spherical2cartesian, cartesian2spherical


def RM_stats(df, title, xlabel, ylabel):
      
    fig, ax = plt.subplots(figsize=(15,3))
    ax2 = ax.twinx()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

#     df['kappa_norm'] = df['k'] / df['k'].max()
#     df['N_norm'] = df['N'] / df['N'].max()

    dfm = df[['age', 'A95', 'n_studies', 'csd']].melt('age', var_name='type', value_name='value')


    sns.lineplot(data  = dfm, x = dfm['age'], y = dfm['value'], hue = dfm['type'], marker="o", ax=ax)
    
    sns.lineplot(data  = df, x = df['age'], y = df['k'], marker="o",  ax=ax2, color= "r")
    ax2.yaxis.label.set_color('red')
    # ax2.legend(handles=[a.lines[0] for a in [ax,ax2]], 
    #        labels=["kappa"])
    
    
def plot_pole_A95(Plat, Plon, A95, age, min_age, max_age, ax):
    """
    Before calling this function set the figure and axes as, for instance:
    
    fig = plt.figure(figsize=(20,10))
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-55) #30, -60
    ax = plt.axes(projection=proj)    
    ax.stock_img()
    ax.set_global() 
    ax.coastlines(linewidth=1, alpha=0.5)
    ax.gridlines(linewidth=1)

    """    
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(min_age, max_age)
        
    ax.add_geometries([Polygon(Geodesic().circle(lon=Plon, lat=Plat, radius=A95*111139, n_samples=360, endpoint=True))], 
                      crs=ccrs.PlateCarree().as_geodetic(), 
                      facecolor='none', 
                      edgecolor=cmap(norm(age)), 
                      alpha=0.6, 
                      linewidth=1.5)
    plt.scatter(x = Plon, y = Plat, color = cmap(norm(age)),
                s=50, transform = ccrs.PlateCarree(), zorder=4,edgecolors='black', alpha = 0.7)
    

def plot_pole(Plat, Plon, age, min_age, max_age, ax):
    """
    Before calling this function set the figure and axes as, for instance:
    
    fig = plt.figure(figsize=(20,10))
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-55) #30, -60
    ax = plt.axes(projection=proj)    
    ax.stock_img()
    ax.set_global() 
    ax.coastlines(linewidth=1, alpha=0.5)
    ax.gridlines(linewidth=1)

    """    
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(min_age, max_age)
        
    plt.scatter(x = Plon, y = Plat, color = cmap(norm(age)),
                s=50, transform = ccrs.PlateCarree(), zorder=4,edgecolors='black', alpha = 0.7)    
    
    

def plot_APWP_df (df_apwp, extent, plot_A95s=True, connect_poles=False):
    """
    Functions to plot an APWP in the form of DF (as spited by the running mean function)
    size scaling is N-dependant
    Input:  a DataFrame with columns ['age','N','n_studies','k','A95','csd','plon','plat']
    """
    
    fig = plt.figure(figsize=(20,10))   
    
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-55) #30, -60

    ax = fig.add_subplot(1,2,1, projection=proj)    
    ax.stock_img()
    ax.coastlines(linewidth=1, alpha=0.5)
    ax.gridlines(linewidth=1)
    
    cmap = mpl.cm.get_cmap('viridis')
    
    # plot the A95s
    if plot_A95s:
        norm = mpl.colors.Normalize(df_apwp["age"].min(), df_apwp["age"].max())
        df_apwp['geom'] = df_apwp.apply(lambda row: Polygon(Geodesic().circle(lon=row["plon"], lat=row["plat"], radius=row["A95"]*111139, n_samples=360, endpoint=True)), axis=1)
        for idx, row in df_apwp.iterrows():
            ax.add_geometries([df_apwp['geom'][idx]], crs=ccrs.PlateCarree().as_geodetic(), facecolor='none', edgecolor=cmap(norm(df_apwp["age"][idx])), 
                              alpha=0.6, linewidth=1.5)
        df_apwp.drop(['geom'], axis=1)

    
    sns.scatterplot(x = df_apwp["plon"], y = df_apwp["plat"], hue = df_apwp["age"], palette=cmap, size = df_apwp["N"], sizes=(50, 200),
                transform = ccrs.PlateCarree(), zorder=4)
    if connect_poles:
        plt.plot(df_apwp["plon"], df_apwp["plat"], transform = ccrs.Geodetic(),  linewidth=1.5)
        
    if extent != 'global':
        ax.set_extent(extent, crs = ccrs.PlateCarree())
        
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(reversed(handles), reversed(labels))
    
def plot_poles_and_APWP(extent, df_poles, df_apwp):
    """
    generates a side by side graphic showing the underlying paleomagnetic poles (in the left) along
    with the Running Means path (in the right plot)
    """
    proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90.0)
    plt.figure(figsize=(12,10))

    ax1 = plt.subplot(221, projection = proj)
    ax1.patch.set_visible(False)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.stock_img()
    ax1.set_extent(extent, crs = ccrs.PlateCarree())
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='-')
    gl.ylabels_left = True

    plt.title('Overview map of Paleomagnetic Poles')
    for _, pole in df_poles.iterrows():
        plot_pole_A95(pole.Plat, pole.Plon, pole.A95, pole.mean_age, df_poles.mean_age.min(), df_poles.mean_age.max(), ax1)
    plt.tight_layout()


    ax2 = plt.subplot(222, projection=proj)
    ax2.set_title('Running Mean path on Paleomagnetic Poles')
    ax2.add_feature(cfeature.BORDERS)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='-')
    gl.ylabels_left = True
    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.stock_img()
    ax2.set_extent(extent, crs = ccrs.PlateCarree())

    for _, pole in df_apwp.iterrows():
        plot_pole_A95(pole.plat, pole.plon, pole.A95, pole.age, df_apwp.age.min(), df_apwp.age.max(), ax2)

    plt.plot(df_apwp["plon"], df_apwp["plat"], transform = ccrs.Geodetic(),  linewidth=1.5, color = "black")
    plt.tight_layout()
    
    s = plt.scatter(
        df_apwp.plon,
        df_apwp.plat,
        c = df_apwp.age,
        edgecolors= "black", marker = "o", s = 25,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    plt.tight_layout()

    plt.colorbar(s, fraction=0.035).set_label("Age (My)")    
    
def RM_APWP_lat_lon_A95 (df_apwp):
    
    df_apwp["plon_"] = df_apwp.apply(lambda row: row.plon - 360 if row.plon > 180 else row.plon, axis =1)
    
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15,6))
    fig.suptitle('Running Mean APWP', fontsize= 16)
    axes[0].set_title('Latitude (°N)', fontsize=12)
    axes[1].set_title('Longitude (°E)', fontsize=12)

    # plot latitude
    axes[0].errorbar(df_apwp["age"].to_list(), df_apwp["plat"].to_list(), yerr=df_apwp["A95"].to_list(),zorder=1) #, fmt="o")
    axes[0].scatter(df_apwp["age"].to_list(), df_apwp["plat"].to_list(), edgecolors = "black",zorder=2)
    axes[0].set_ylabel(r'Latitude (°N)', fontweight ='bold')
    # axes[0].set_ylabel(-90,-75)
    
    # plot longitude    
    axes[1].errorbar(df_apwp["age"].to_list(), df_apwp["plon_"].to_list(), yerr=df_apwp["A95"].to_list(),zorder=1)#, fmt="o")
    axes[1].scatter(df_apwp["age"].to_list(), df_apwp["plon_"].to_list(), edgecolors = "black",zorder=2)   
    axes[1].set_xlabel(r'Age (Ma)', fontweight ='bold')
    axes[1].set_ylabel(r'Longitude (°E)', fontweight ='bold')
    
    
    del df_apwp["plon_"]

def plot_pseudoVGPs_and_APWP(extent, df_vgps, df_apwp):
    """
    generates a side by side graphic showing the underlying paleomagnetic poles (in the left) along
    with the Running Means path (in the right plot)
    """
    proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90.0)
    plt.figure(figsize=(12,10))

    ax1 = plt.subplot(221, projection = proj)
    ax1.patch.set_visible(False)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.stock_img()
    #ax1.set_extent(extent, crs = ccrs.PlateCarree())
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='-')
    gl.ylabels_left = True

    plt.title('Overview map of parametrically sampled VGPs ($pseudo$-VGPs)')
    
    for _, pole in df_vgps.iterrows():
        plot_pole(pole.Plat, pole.Plon, pole.mean_age, df_vgps.mean_age.min(), df_vgps.mean_age.max(), ax1)
    plt.tight_layout()


    ax2 = plt.subplot(222, projection=proj)
    ax2.set_title('Running Mean path on $pseudo$-VGPs')
    ax2.add_feature(cfeature.BORDERS)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='-')
    gl.ylabels_left = True
    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.stock_img()
    ax2.set_extent(extent, crs = ccrs.PlateCarree())

    for _, pole in df_apwp.iterrows():
        plot_pole_A95(pole.plat, pole.plon, pole.A95, pole.age, df_apwp.age.min(), df_apwp.age.max(), ax2)

    plt.plot(df_apwp["plon"], df_apwp["plat"], transform = ccrs.Geodetic(), linewidth=1.5, color = "black")
    plt.tight_layout()
    
    s = plt.scatter(
        df_apwp.plon,
        df_apwp.plat,
        c = df_apwp.age,
        edgecolors= "black", marker = "o", s = 25,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    plt.tight_layout()

    plt.colorbar(s, fraction=0.035).set_label("Age (My)")  
    
def plot_VGPs_and_APWP(extent, df_vgps, df_apwp):
    """
    generates a side by side graphic showing the underlying VGPs along
    with the Running Means path (in the right plot)
    """
    proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90.0)
    plt.figure(figsize=(12,10))

    ax1 = plt.subplot(221, projection = proj)
    ax1.patch.set_visible(False)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.stock_img()
    #ax1.set_extent(extent, crs = ccrs.PlateCarree())
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='-')
    gl.ylabels_left = True

    plt.title('Overview map of recompiled VGPs')
    
    for _, pole in df_vgps.iterrows():
        plot_pole(pole.vgp_lat_SH, pole.vgp_lon_SH, pole.mean_age, df_vgps.mean_age.min(), df_vgps.mean_age.max(), ax1)
    plt.tight_layout()


    ax2 = plt.subplot(222, projection=proj)
    ax2.set_title('Running Mean path on VGPs')
    ax2.add_feature(cfeature.BORDERS)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='-')
    gl.ylabels_left = True
    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.stock_img()
    ax2.set_extent(extent, crs = ccrs.PlateCarree())

    for _, pole in df_apwp.iterrows():
        plot_pole_A95(pole.plat, pole.plon, pole.A95, pole.age, df_apwp.age.min(), df_apwp.age.max(), ax2)

    plt.plot(df_apwp["plon"], df_apwp["plat"], transform = ccrs.Geodetic(), linewidth=1.5, color = "black")
    plt.tight_layout()
    
    s = plt.scatter(
        df_apwp.plon,
        df_apwp.plat,
        c = df_apwp.age,
        edgecolors= "black", marker = "o", s = 25,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    plt.tight_layout()

    plt.colorbar(s, fraction=0.035).set_label("Age (My)") 

def plot_APWP_RM_ensemble(df, title):
    '''
    pass a df with the colection of bootstrapped means for each run.
    '''
    df['plon_east'] = df.apply(lambda row: row.plon - 360 if row.plon > 180 else row.plon, axis =1)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15,6))
    fig.suptitle(title, fontsize= 16, fontweight ='bold')
    axes[0].set_title('Latitude (°N)', fontsize=12)
    axes[1].set_title('Longitude (°E)', fontsize=12)

    # plot latitude
    pltt = sns.scatterplot(data=df, x="age", y="plat", ax = axes[0])
    axes[0].set_ylabel(r'Latitude (°N)', fontweight ='bold')
    # plot longitude
    pltt1 = sns.scatterplot(data=df, x="age", y= df['plon_east'], ax = axes[1])
    axes[1].set_ylabel(r'Longitude (°E)', fontweight ='bold')
    axes[1].set_xlabel(r'Age (Ma)', fontweight ='bold')
    
    x = sorted(df['age'].unique())
    Y = df.groupby('age')['plat']

    q5 =  Y.quantile(.16).to_numpy() 
    q25 = Y.quantile(.25).to_numpy()
    q50 = Y.quantile(.50).to_numpy()
    q75 = Y.quantile(.75).to_numpy() 
    q95 = Y.quantile(.84).to_numpy() 
    mean = Y.mean().to_numpy()
    pltt.fill_between(x, q5,q95, color= "#71AFE2", alpha=.50,label="0.16-0.84 percentiles")
    pltt.fill_between(x, q25,q75, color= "#1A52C1", alpha=.50,label="0.25-0.75 percentiles")
    pltt.plot(x, mean, '--', color="#ad3131",label="mean")
   
    x_ = sorted(df['age'].unique())
    Y_ = df.groupby('age')['plon_east']

    q5_ =  Y_.quantile(.05).to_numpy() 
    q25_ = Y_.quantile(.25).to_numpy()
    q50_ = Y_.quantile(.50).to_numpy()
    q75_ = Y_.quantile(.75).to_numpy() 
    q95_ = Y_.quantile(.95).to_numpy() 
    mean_ = Y_.mean().to_numpy()

    pltt1.fill_between(x_, q5_,q95_, color= "#71AFE2", alpha=.50,label="0.16-0.84 percentiles")
    pltt1.fill_between(x_, q25_,q75_, color= "#1A52C1", alpha=.50,label="0.25-0.75 percentiles")
    pltt1.plot(x_, mean_, '--', color="#ad3131",label="mean")
    
    df = df.drop(['plon_east'], axis=1)
    
class quantiles:
    '''
    class to generate quantiles.
    note: input fro longitudes should live in [-180,180]
    '''
    
    def __init__(self, df, xlabel, ylabel):        
        self.X = df[xlabel].unique().transpose()        
        self.Y = df.groupby(xlabel)[ylabel]
        
        self.q5 = self.Y.quantile(.05).to_numpy()
        self.q16 = self.Y.quantile(.16).to_numpy()
        self.q25 = self.Y.quantile(.25).to_numpy()
        self.q50 = self.Y.quantile(.50).to_numpy()
        self.q75 = self.Y.quantile(.75).to_numpy()
        self.q84 = self.Y.quantile(.84).to_numpy()
        self.q95 = self.Y.quantile(.95).to_numpy()
        self.mean = self.Y.mean().to_numpy()
    
class PC:
    
    '''
    Class to calculate PCs as a function of time from a time dependant ensemble of directions
    '''
    def __init__(self, df, xlabel, LatLabel, LonLabel):
        
        self.X = df[xlabel].unique().transpose()
        self.df=df
        self.xlabel=xlabel
        self.LatLabel=LatLabel
        self.LonLabel=LonLabel
        self.groupby=df.groupby(xlabel)
        
    def PC(self):
        lats, lons = [], []
        for age, df_age in self.groupby:
            array = np.array([spherical2cartesian([ np.radians(i[self.LonLabel]),np.radians(i[self.LatLabel])]) for _,i in df_age.iterrows()])
            eigenValues, eigenVectors = eigen_decomposition(array)
            lats.append(np.degrees(cartesian2spherical(eigenVectors[:,0]))[1])
            lons.append(np.degrees(cartesian2spherical(eigenVectors[:,0]))[0])
            
#             print(np.degrees(cartesian2spherical(eigenVectors[:,0])), age, len(array))
        
        return [np.array(lons),np.array(lats)]