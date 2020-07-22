# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:17:18 2020

@author: mathewjowens
"""

#a script to produce a synthetic HI image from HelioMAS output
#(Mathew Owens, 30/6/2020)

import os
import glob
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import spherical_to_cartesian
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import cKDTree

mpl.rc("axes", labelsize=14)
mpl.rc("ytick", labelsize=13)
mpl.rc("xtick", labelsize=13)
mpl.rc("legend", fontsize=13)

    
# <codecell>  Code from Luke's CME_Ghosts script

# Define lines of sight for a range of elongation angles
class GetSun:
    
    def __init__(self):
        self.x = 0.0 * u.m
        self.y = 0.0 * u.m
        self.radius = const.R_sun
        self.B0 = 2.3e7 * (u.W / (u.m**2 * u.steradian))
        
class Observer:
    
    def __init__(self, radius, longitude):
        
        # Position of the observer
        self.r = radius * u.AU
        self.r = self.r.to(u.m)
        self.longitude = np.deg2rad(longitude) * u.rad
        self.x = self.r * np.cos(self.longitude)
        self.y = self.r * np.sin(self.longitude)
        
        
class Imager:
    
    def __init__(self, observer, modelrho, rho_r, rho_long, rho_lat):
        
        #get the max and min radial distances
        self.rmin=np.min(rho_r)
        self.rmax=np.max(rho_r)
        
        #first create the meshed grid for the density data cube
        grid_long, grid_lat, grid_r = np.meshgrid(rho_long, rho_lat, rho_r, indexing='ij')
        #convert to x,y,z
        self.grid_x, self.grid_y, self.grid_z = spherical_to_cartesian(grid_r.to(u.m), 
                                                                       grid_lat, 
                                                                       grid_long)
        self.modelrho=modelrho
        
        
        # Position of the imager is defined by the observer
        self.position = observer
        # Rectify the longitude so it runs from 0 > 360.
        if self.position.longitude < 0:
            self.position.longitude = 2.0*np.pi*u.rad + self.position.longitude
        
        # Define the fied of view in terms of elongation and line of sight distance
        # Elongations
        self.nelon = 100
        e_min = np.deg2rad(8.0)
        e_max = np.deg2rad(40.0)
        self.elon, self.de = np.linspace(e_min, e_max, self.nelon, retstep=True)
        self.elon = self.elon * u.rad
        self.de = self.de * u.rad

        # LOS distance
        self.nL = 2000
        L_min = 0.0
        L_max = 2.0 * u.AU.to(u.m)
        self.L, self.dL = np.linspace(L_min, L_max, self.nL, retstep=True)
        self.L = self.L * u.m
        self.dL = self.dL * u.m
        
        
        # Mesh these two coordinates, to compute other required parameters:
        self.elon_grid, self.L_grid = np.meshgrid(self.elon, self.L)
        
        # Heliocentric distance of each LOS element - using cosine rule
        B = self.position.r
        C = self.L_grid
        self.r = np.sqrt(B**2 + C**2 - (2.0 * B * C * np.cos(self.elon_grid)))
        # Scattering angle of radial illumination at each LOS - using cosine rule
        self.chi = np.arccos((C**2 + self.r**2 - B**2) / (2.0 * C * self.r))
        # Find the angle that completes the observer-scattering site-sun triangle
        theta_ls = np.pi * u.rad - self.elon_grid - self.chi
        # Cartesian coord of each LOS blob
        # This appears to work. Should be careful to check FOV when thinking about STA or STB
        # particularly for lon near 0 or 180.
        if self.position.longitude < np.pi*u.rad:
            phi_ls = self.position.longitude - theta_ls
        else:
            phi_ls = theta_ls - (2*np.pi*u.rad - self.position.longitude)
            
        self.x = self.r * np.cos(phi_ls) 
        self.y = self.r * np.sin(phi_ls) 
        self.z = self.r * 0.0 
        
        # Arc length half way though each LOS element
        self.ds = (self.L_grid + self.dL/2.0) * self.de.value
        # Cross section of each element
        self.dA = self.ds*self.ds
        # Volume of each element
        self.dV = self.dA * self.dL
        
        # Compute the elongation of the CME nose and tangents
        #self._compute_cme_elongation_(cme)
        # Compute the solar wind density at each LOS element
        self.rho = self._compute_los_density_()
        self.dI, self.I = self._compute_intensity_()
                
        
    def _compute_los_density_(self):
        """
        Function to compute the density of the solar wind 
        """
        rho_ls=np.ones(self.x.shape)
       
        rho_ls=interp3d(self.x, self.y, self.z,
                        self.modelrho, 
                        self.grid_x, self.grid_y, self.grid_z, 
                        n_neighbour = 6)
       
        return rho_ls                
                
    def _compute_intensity_(self):
        """
        Compute the Thomson scattered intensity throughout the FOV defined by the lines of sight. 
        """        
        # Compute Thomson scatter cross section as per Howard and DeForest 2012.
        re = (1.0 / (4.0*np.pi*const.eps0)) * (const.e.si**2 / (const.m_e * const.c**2))
        sigma_t = re**2 / 2.0
        sigma_t = sigma_t.to(u.m**2)
        # Solar surface brightness, as per Howard and DeForest 2012.
        Bo = 2.3e7 * (u.W / (u.m**2 * u.steradian))
        solar_radius = const.R_sun.value * const.R_sun.unit
        
        # Compute differential intensity of parcel (eqn 9 of HD12)
        numerator = np.sin(self.chi)**4 * (1.0 + np.cos(self.chi)**2)
        denominator = (np.sin(self.elon)*np.sin(self.elon + self.chi))**2
        dI = (Bo * sigma_t * np.pi * solar_radius**2 / self.r**4) * (numerator / denominator) * self.rho * self.dV
        I = np.nansum(dI, axis=0)
        return dI, I
    
def plot_frame(i, t, sun, obs, img, savefig=False, tag="HI_frame",fig_dir="D:\\Dropbox\\python_repos\\HI_simulator\\Figures"):
    

    #logrhomin=-8.0; logrhomax=-4.5
    #logdemin=-28; logdemax=-18
    
    logrhomin=np.nanmin(np.log10(img.modelrho))
    logrhomax=np.nanmax(np.log10(img.modelrho))
    
    logdemin=np.nanmin(np.log10(img.dI.value))
    logdemax=np.nanmax(np.log10(img.dI.value))
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.6))
    levels = np.arange(logrhomin, logrhomax, 0.2)
    #cnt = ax[0].contourf(img.x.to('AU'), img.y.to('AU'), np.log10(img.rho), levels=levels)
    
    
    
    #here, the code has been modified to use 2D HUXt
    #i_eq=np.argmin(np.absolute(img.grid_z[0,:,0]))
    ax[0].scatter(img.grid_x.to('AU'),img.grid_y.to('AU'),
                  c=np.log10(img.modelrho), norm=plt.Normalize(logrhomin,logrhomax))
    
    
    
    
    
    
    cnt = ax[0].contourf(img.x.to('AU'), img.y.to('AU'), np.log10(img.rho), levels=levels)
    ax[0].plot(sun.x.to('AU'), sun.y.to('AU'), 'yo', markersize=10, label='Sun')
    ax[0].plot(img.position.x.to('AU'), img.position.y.to('AU'), 'co', markersize=10, label='STA')

     # Find LOS closest to the elon of the nose and the forward tangent
    id_tangent = 0
    id_nose = img.nelon - 1
    for id_feature, label, color in zip([id_tangent, id_nose], ['Inner', 'Outer'], ['k', 'm']):
            ax[0].plot(img.x[:, id_feature].to('AU'), img.y[:, id_feature].to('AU'), '-', color=color, label=label)
            ax[1].plot(img.L.to('AU'), np.log10(img.rho[:, id_feature]), '-', color=color, label=label)
            ax[2].plot(img.L.to('AU'), np.log10(img.dI[:, id_feature].value), '-', color=color, label=label)
   
    ax[0].set_xlim(-1.0, 1.0)        
    ax[0].set_ylim(-1.0, 1.0)        
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('X (AU)')
    ax[0].set_ylabel('Y (AU)')

    ax[1].set_xlim(0, 2.0)        
    #ax[1].set_ylim(1e6, 1e11)        
    ax[1].set_xlabel('Line of sight distance (AU)')
    #ax[1].set_ylabel('Electorn number density ($m^{-3}$) [mode units]')
    ax[1].set_ylim(logrhomin, logrhomax)
    ax[1].set_ylabel('Log (Electorn number density) [model units]')
    
    ax[2].set_xlim(0, 2.0)        
    #ax[2].set_ylim(1e-15, 1e-4)  
    ax[2].set_ylim(logdemin, logdemax)         
    ax[2].set_xlabel('Line of sight distance (AU)')
    ax[2].set_ylabel('Log (Differential intensity) [model units]')

       
    for a in ax:
        a.legend(loc='upper left')

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.9, wspace=0.2)

    # Add in the colorbar
    pos = ax[0].get_position()
    dw = 0.005
    dh = 0.005
    left = pos.x0 + dw
    bottom = pos.y1 + dh
    wid = pos.width - 2 * dw
    cnt_cbaxes = fig.add_axes([left, bottom, wid, 0.015])
    cbar1 = fig.colorbar(cnt, cax=cnt_cbaxes, orientation='horizontal')
    cbar1.ax.set_xlabel("Log [Electorn number density ($m^{-3}$)]")
    cbar1.ax.xaxis.tick_top()
    cbar1.ax.xaxis.set_label_position('top')
    if savefig:
        fig_name = "{}_t{:02d}.png".format(tag, i)
        #fig_dir = "C:\\Users\\yq904481\\PyCharmProjects\\cme_ghosts\\figures"
        #fig_dir = "D:\\Dropbox\\python_repos\\CME_Ghosts\\Figures"
        fig_path = os.path.join(fig_dir, fig_name)
        fig.savefig(fig_path)
        plt.close('all')

    return

def make_animation(src, dst, tidy=True):
    cmd = " ".join(["magick convert -delay 5 -loop 0 ", src, dst])
    os.system(cmd)
    if tidy:
        files = glob.glob(src)
        for f in files:
            os.remove(f)    

# <codecell> interp3d
def interp3d(xi, yi, zi, V, x, y, z, n_neighbour = 4):
    """
    Fast 3d interpolation on an irregular grid. Uses the K-Dimensional Tree
    implementation in SciPy. Neighbours are weighted by 1/d^2, where d is the 
    distance from the required point.
    
    Based on Earthpy exmaple: http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    
    Mathew Owens, 8/7/20

    Parameters
    ----------
    xi, yi, zi :  Ni x Mi arrays of new positions at which to interpolate. 
    
    V : N x M array of the parameter field to be interpolated
    
    x, y, z: N x M arrays of the position of the parameter field, V
        
    n_neighbour : Number of neighbours to use in interpolation. The default is 4.

    Returns
    -------
    Vi : Ni x Mi array of the parameter at new positions.

    """
    
    #check that the dimensions of the coords and V are the same
    assert len(V) == len(x)
    assert len(x) == len(y)
    assert len(y) == len(z)
    assert len(xi) == len(yi)
    assert len(yi) == len(zi)
    
    
    #create a list of grid points
    gridpoints=np.ones((len(x.flatten()),3))
    gridpoints[:,0]=x.flatten()
    gridpoints[:,1]=y.flatten()
    gridpoints[:,2]=z.flatten()
    
    #create a list of densities
    V_list=V.flatten()

    #Create cKDTree object to represent source grid
    tree=cKDTree(gridpoints)
    
    #get the size of the new coords
    origsize=xi.shape

    newgridpoints=np.ones((len(xi.flatten()),3))
    newgridpoints[:,0]=xi.flatten()
    newgridpoints[:,1]=yi.flatten()
    newgridpoints[:,2]=zi.flatten()
    
    #nearest neighbour
    #d, inds = tree.query(newgridpoints, k = 1)
    #rho_ls[:,ie]=rholist[inds]
    
    #weighted sum of N nearest points
    distance, index = tree.query(newgridpoints, k = n_neighbour)
    #tree.query  will sometimes return an index past the end of the grid list?
    index[index>=len(gridpoints)]=len(gridpoints)-1
    
    #weight each point by 1/dist^2
    weights = 1.0 / distance**2
    
    #generate the new value as the weighted average of the neighbours
    Vi_list = np.sum(weights * V_list[index], axis=1) / np.sum(weights, axis=1)
    
    #revert to original size
    Vi=Vi_list.reshape(origsize)
    
    return Vi

# <codecell> Jmap generation from HUXt model

def Jmap_HUXt(model, obs= Observer(1.0, 90.0)):

    time = model.time_out
    #determine brightness along LOS in the equatorial plane
    for i, t in enumerate(time):
        print('Processing HI frame ' +str(i+1) + ' of '+str(time.size))
        
        # #rotate the solution
        # Tsid=2192832
        # spinlong=rho_long + 2*np.pi *u.rad * t / Tsid
        rho=np.flipud(np.rot90(model.rho_grid[i,:,:]))
        
        img = Imager(obs, rho, model.r, model.lon, model.latitude)
        if i == 0:
            jmap = np.zeros((img.I.size, time.size))
            
        jmap[:, i] = img.I.value
        
    # pickle_out = open("jmap.pickle","wb")
    # pickle.dump(jmap, pickle_out)
    # pickle_out.close()
    # jmap = pickle.load( open( "jmap.pickle", "rb" ) )
    

    
    mymap = mpl.cm.cividis
    mymap.set_over([1, 1, 1])
    mymap.set_under([0, 0, 0])
    
    
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax = [ax1, ax2]
    
    elon = img.elon.to('deg').value
    fov = (elon >= 10.0) & (elon <= 40.0)
    time = model.time_out.to(u.day).value
    
    dj = jmap[fov, 1:] - jmap[fov, 0:-1]
    
    ax[0].pcolor(time, elon[fov], jmap[fov, :],vmin=0,vmax=1e-14,cmap='gray')
    #ax[0].text(0.05, 0.9, 'Absolute brightness', fontsize=16, transform=ax[1].transAxes, color='white')
    ax[0].set_ylabel('Elongation [degrees]')
    ax[0].set_xlabel('Time [days]')
    
    ax[1].pcolor(time, elon[fov], dj,vmin=-1e-16,vmax=1e-16,cmap='gray')
    #ax[1].text(0.05, 0.9, 'Running difference', fontsize=16, transform=ax[1].transAxes, color='white')
    ax[1].set_ylabel('Elongation [degrees]')
    ax[1].set_xlabel('Time [days]')





