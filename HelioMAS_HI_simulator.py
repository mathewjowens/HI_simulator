# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:17:18 2020

@author: mathewjowens
"""

#a script to produce a synthetic HI image from HelioMAS output
#(Mathew Owens, 30/6/2020)
import httplib2
import urllib
import os
import glob
from pyhdf.SD import SD, SDC  
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import spherical_to_cartesian
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
from scipy.spatial import cKDTree
import pickle

mpl.rc("axes", labelsize=14)
mpl.rc("ytick", labelsize=13)
mpl.rc("xtick", labelsize=13)
mpl.rc("legend", fontsize=13)

# <codecell> Load HelioMAS
def getMASsolution(cr=np.NaN, param='rho', observatory='', runtype='', runnumber=''):
    """
    A function to grab the  Vr and Br boundary conditions from MHDweb. An order
    of preference for observatories is given in the function. Checks first if
    the data already exists in the HUXt boundary condition folder

    Parameters
    ----------
    cr : INT
        Carrington rotation number 
    param : STR
        Name of parameter to download
    observatory : STRING
        Name of preferred observatory (e.g., 'hmi','mdi','solis',
        'gong','mwo','wso','kpo'). Empty if no preference and automatically selected 
    runtype : STRING
        Name of preferred MAS run type (e.g., 'mas','mast','masp').
        Empty if no preference and automatically selected 
    runnumber : STRING
        Name of preferred MAS run number (e.g., '0101','0201').
        Empty if no preference and automatically selected    

    Returns
    -------
    flag : INT
        1 = successful download. 0 = files exist, -1 = no file found.

    """
    
    assert(np.isnan(cr)==False)
    
    #the order of preference for different MAS run results
    overwrite=False
    if not observatory:
        observatories_order=['hmi','mdi','solis','gong','mwo','wso','kpo']
    else:
        observatories_order=[str(observatory)]
        overwrite=True #if the user wants a specific observatory, overwrite what's already downloaded
        
    if not runtype:
        runtype_order=['masp','mas','mast']
    else:
        runtype_order=[str(runtype)]
        overwrite=True
    
    if not runnumber:
        runnumber_order=['0201','0101']
    else:
        runnumber_order=[str(runnumber)]
        overwrite=True
    

      
    #example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/rho002.hdf 
    heliomas_url_front='http://www.predsci.com/data/runs/cr'
    heliomas_url_end='002.hdf'
    
    outfilename = 'HelioMAS_CR'+str(int(cr)) + '_'+param +heliomas_url_end
    
    if (os.path.exists(outfilename) == False or  overwrite==True): #check if the files already exist
        #Search MHDweb for a HelioMAS run, in order of preference 
        h = httplib2.Http()
        foundfile=False
        for masob in observatories_order:
            for masrun in runtype_order:
                for masnum in runnumber_order:
                    urlbase=(heliomas_url_front + str(int(cr)) + '-medium/' + masob +'_' +
                         masrun + '_mas_std_' + masnum + '/helio/')
                    url=urlbase + param + heliomas_url_end
                    #print(url)
                    
                    #see if this br file exists
                    resp = h.request(url, 'HEAD')
                    if int(resp[0]['status']) < 400:
                        foundfile=True
                        #print(url)
                    
                    #exit all the loops - clumsy, but works
                    if foundfile: 
                        break
                if foundfile:
                    break
            if foundfile:
                break
            
        if foundfile==False:
            print('No data available for given CR and observatory preferences')
            return -1
        
        #download teh vr and br files            
        print('Downloading from: ',urlbase)
        urllib.request.urlretrieve(urlbase + param + heliomas_url_end,
                           os.path.join(outfilename) )    

        
        return 1
    else:
         print('Files already exist for CR' + str(int(cr)))   
         return 0
     
        
def readMASparam(cr, param='rho'):
    """
    A function to read in the MAS coundary conditions for a given CR

    Parameters
    ----------
    cr : INT
        Carrington rotation number
    param : STR
        Name of parameter to load
        
    Returns
    -------
    MAS_vr : NP ARRAY (NDIM = 2)
        Solar wind speed at 30rS, in km/s
    MAS_vr_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Vr map, in rad
    MAS_vr_Xm : NP ARRAY (NDIM = 1)
        Latitude of Vr as angle down from N pole, in rad
    MAS_br : NP ARRAY (NDIM = 2)
        Radial magnetic field at 30rS, in model units
    MAS_br_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Br map, in rad
    MAS_br_Xm : NP ARRAY (NDIM = 1)
       Latitude of Br as angle down from N pole, in rad

    """

    #create the filenames 
    heliomas_url_end='002.hdf'
    filename = 'HelioMAS_CR'+str(int(cr)) + '_' + param + heliomas_url_end
    
    assert os.path.exists(filename)
    #print(os.path.exists(filepath))

    file = SD(filename, SDC.READ)
    # print(file.info())
    # datasets_dic = file.datasets()
    # for idx,sds in enumerate(datasets_dic.keys()):
    #     print(idx,sds)
        
    sds_obj = file.select('fakeDim0') # select sds
    MAS_vr_Xa = sds_obj.get() # get sds data
    sds_obj = file.select('fakeDim1') # select sds
    MAS_vr_Xm = sds_obj.get() # get sds data
    sds_obj = file.select('fakeDim2') # select sds
    MAS_vr_Xr = sds_obj.get() # get sds data
    sds_obj = file.select('Data-Set-2') # select sds
    MAS_vr = sds_obj.get() # get sds data
    
    
    if param == 'vr' or param =='vp' or param =='vt':
        #convert from model to physicsal units
        MAS_vr = MAS_vr*481.0 * u.km/u.s
        
    MAS_vr_Xa=MAS_vr_Xa * u.rad
    MAS_vr_Xm=MAS_vr_Xm * u.rad
    MAS_vr_Xr=MAS_vr_Xr * u.solRad
    
    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_vr_Xr


cr=2140
getMASsolution(cr=cr, param = 'rho')
MAS_rho, MAS_rho_Xa, MAS_rho_Xm, MAS_rho_Xr = readMASparam(cr, param='rho')

rho=MAS_rho
rho_r=MAS_rho_Xr
rho_long=MAS_rho_Xa
#convert from angle from Npole to angle from equator
rho_lat= (np.pi/2)*u.rad - MAS_rho_Xm
#flip lats, so they're increasing in value
rho_lat=np.flipud(rho_lat)
#flip the daat cube accordingly
for r in range(0,len(rho_r)):
    rho[:,:,r]=np.fliplr(rho[:,:,r])
    
    

    
#HelioMAS rho appears to run from -91 to +91 latitude, presumably as a result of a staggered grid with V    
#Ditch the first and last values
L=len(rho_lat)
rho_lat=np.delete(rho_lat,[0,L-1])
rho=np.delete(rho,[0,L-1],axis=1)


    
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
        e_max = np.deg2rad(48.0)
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
        # for ie in range(0,self.nelon):
        #     print('Elongation '+str(ie+1) + ' of ' +str(self.nelon) )
        #     for iL in range(0,self.nL):
        #         #the next point alongthe LOS
        #         p_x = self.x[iL,ie]
        #         p_y = self.y[iL,ie]
        #         p_z = self.z[iL,ie]
        #         p_r = np.sqrt(p_x*p_x + p_y*p_y + p_z*p_z)
                
        #         if p_r <= self.rmax and p_r <= self.rmax :
        #             #compute the distance from every point in the grid
        #             nmin=nearest_neighbour_index(p_x, p_y, p_z, self.grid_x, self.grid_y, self.grid_z)
                    
        #             imin=np.unravel_index(nmin, self.modelrho.shape)
        #             #find the nearest rho value to a given point
        #             rho_ls[iL,ie]=self.modelrho[imin]
        #         else:
        #             rho_ls[iL,ie]=np.nan
        
        
        #create a list of grid points
        gridpoints=np.ones((len(self.grid_x.flatten()),3))
        gridpoints[:,0]=self.grid_x.flatten()
        gridpoints[:,1]=self.grid_y.flatten()
        gridpoints[:,2]=self.grid_z.flatten()
        
        #create a list of densities
        rholist=self.modelrho.flatten()
        
        #Create cKDTree object to represent source grid
        tree=cKDTree(gridpoints)
        
        for ie in range(0,self.nelon):
            #create a list of all the required new points
            xs=self.x[:,ie].flatten()
            ys=self.y[:,ie].flatten()
            zs=self.z[:,ie].flatten()
            newgridpoints=np.ones((len(xs),3))
            newgridpoints[:,0]=xs
            newgridpoints[:,1]=ys
            newgridpoints[:,2]=zs
            
            #nearest neighbour
            #d, inds = tree.query(newgridpoints, k = 1)
            #rho_ls[:,ie]=rholist[inds]
            
            #weighted sum of 4 nearest points
            d, inds = tree.query(newgridpoints, k = 4)
            inds[inds>=len(gridpoints)]=len(gridpoints)-1
            w = 1.0 / d**2
            rho_ls[:,ie] = np.sum(w * rholist[inds], axis=1) / np.sum(w, axis=1)
            
            #check points aren't outide the model domain
            #p_r = np.sqrt(xs*xs + ys*ys + zs*zs)
            #rho_ls[p_r > self.rmax] = 0.0
            
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
    
def plot_frame(i, t, sun, obs, img, savefig=False, tag="ghost_frame"):
    

    logrhomin=-8.0; logrhomax=-4.5
    logdemin=-28; logdemax=-18
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.6))
    levels = np.arange(logrhomin, logrhomax, 0.2)
    #cnt = ax[0].contourf(img.x.to('AU'), img.y.to('AU'), np.log10(img.rho), levels=levels)
    i_eq=np.argmin(np.absolute(img.grid_z[0,:,0]))
    ax[0].scatter(img.grid_x[:,i_eq,:].to('AU'),img.grid_y[:,i_eq,:].to('AU'),
                  c=np.log10(img.modelrho[:,i_eq,:]), norm=plt.Normalize(logrhomin,logrhomax))
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
        fig_dir = "D:\\Dropbox\\python_repos\\CME_Ghosts\\Figures"
        fig_path = os.path.join(fig_dir, fig_name)
        fig.savefig(fig_path)
        plt.close('all')

    return

def make_animation(src, dst, tidy=True):
    cmd = "magick ".join(["convert -delay 5 -loop 0 ", src, dst])
    os.system(cmd)
    if tidy:
        files = glob.glob(src)
        for f in files:
            os.remove(f)    

# <codecell>
@jit(nopython=True)
def nearest_neighbour_index(p_x, p_y, p_z, grid_x, grid_y, grid_z):
    #check dimensions of input
    #assert(V.shape == grid_x.shape)
    #assert(V.shape == grid_y.shape)
    #assert(V.shape == grid_z.shape)
    
    #compute the distance from every point in the grid
    dx = grid_x.flatten() - p_x
    dy = grid_y.flatten() - p_y
    dz = grid_z.flatten() - p_z
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    

            
    #find the closest point
    nmin=np.argmin(dist)
    #imin=np.unravel_index(nmin, dist.shape)
    #find the nearest rho value to a given point
    #v_p=V[imin]
    
    return nmin
    





# #def nearest_neighbour_index_KDTree()
# gridpoints=np.ones((len(grid_x.flatten()),3))
# gridpoints[:,0]=grid_x.flatten()
# gridpoints[:,1]=grid_y.flatten()
# gridpoints[:,2]=grid_z.flatten()

# tree=cKDTree(gridpoints)

# xs=img.x.flatten()
# ys=img.y.flatten()
# zs=img.z.flatten()
# newgridpoints=np.ones((len(xs),3))
# newgridpoints[:,0]=xs
# newgridpoints[:,1]=ys
# newgridpoints[:,2]=zs

# d, inds = tree.query(newgridpoints, k = 1)

# <codecell>
sun = GetSun()
obs = Observer(0.976, 47.0) # STA, R(AU), long(deg)
img = Imager(obs, rho, rho_r, rho_long, rho_lat)

#find the equatorial plane and plot
i_eq=np.argmin(np.absolute(img.grid_z[0,:,0]))
fig,ax=plt.subplots(figsize=(10,10))
p=ax.scatter(img.grid_x[:,i_eq,:],img.grid_y[:,i_eq,:],c=img.modelrho[:,i_eq,:])
plt.scatter(img.x,img.y,c=img.rho)



time = np.arange(0, 5.2*24*60*60, 240*60)
# Find frame index to save data
tid = np.argmin(np.abs(time - 2*86400))


#generate movie of equatorial plane, the electron density along LOS and brightness along LOS
for i, t in enumerate(time):
    
    #rotate the solution
    Tsid=2192832
    spinlong=rho_long + 2*np.pi *u.rad * t / Tsid
    
    img = Imager(obs, rho, rho_r, spinlong, rho_lat)
    plot_frame(i, t, sun, obs, img, savefig=True, tag="stfc_ghost_frame_")
    if i == 0:
        jmap = np.zeros((img.I.size, time.size))
        
    jmap[:, i] = img.I.value
    
    if i == tid:
        pickle_out = open("image.pickle","wb")
        pickle.dump(img, pickle_out)
        pickle_out.close()
        
pickle_out = open("jmap.pickle","wb")
pickle.dump(jmap, pickle_out)
pickle_out.close()

fig_dir = "D:\\Dropbox\\python_repos\\CME_Ghosts\\Figures"
out_name = "stfc_ghost_frame__t*.png"
src = os.path.join(fig_dir, out_name)
gif_name = "ani.gif"
dst = os.path.join(fig_dir, gif_name)
make_animation(src, dst, tidy=False)

# <codecell>

sun = GetSun()
obs = Observer(0.976, 47.0) # STA, R(AU), long(deg)

#generate jmap
time = np.arange(0, 10*24*60*60, 240*60)
for i, t in enumerate(time):
    
    #rotate the solution
    Tsid=2192832
    spinlong=rho_long + 2*np.pi *u.rad * t / Tsid
    
    img = Imager(obs, rho, rho_r, spinlong, rho_lat)
    
    if i == 0:
        jmap = np.zeros((img.I.size, time.size))
        
    jmap[:, i] = img.I.value
    
       
pickle_out = open("jmap.pickle","wb")
pickle.dump(jmap, pickle_out)
pickle_out.close()



def fov_coords(rs, ls, elon):
    xs = rs*np.cos(ls)
    ys = rs*np.sin(ls)

    e = np.deg2rad(elon)
    rf = rs*np.sin(e)

    if ls < np.pi/2:
        beta = -((np.pi/2) - e - ls)
    else:
        beta = -1.5*np.pi - e + ls
            
    xf = rf*np.cos(beta)
    yf = rf*np.sin(beta)


    m = (yf - ys) / (xf - xs)
    c = ys - m*xs

    xo = -220
    yo = m*xo + c

    ro = np.sqrt(xo**2 + yo**2)
    lo = np.arctan2(yo, xo)
    
    return [[ls, lo], [rs, ro]]

img = pickle.load( open( "image.pickle", "rb" ) )
jmap = pickle.load( open( "jmap.pickle", "rb" ) )

mpl.rc("axes", labelsize=14)
mpl.rc("ytick", labelsize=14)
mpl.rc("xtick", labelsize=14)
mpl.rc("legend", fontsize=14)

mymap = mpl.cm.cividis
mymap.set_over([1, 1, 1])
mymap.set_under([0, 0, 0])
dv = 10
#fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})

                   
# r = np.arange(30,230,1)
# lon = np.linspace(0,2*np.pi,360)
# lon, r = np.meshgrid(lon, r)
# x = r*np.cos(lon)
# y = r*np.sin(lon)
# rho = CMEG.solar_wind_density(r*u.solRad.to(u.m)*u.m)
# cnt = ax[0].contourf(lon, r, np.log10(rho.value), levels, cmap=mymap)
# lon = np.arctan2(img.y.value, img.x.value)
# rho = img.rho.value
# id_mask = img.r.to('solRad').value < 30
# rho[id_mask] = np.NaN
# cnt2 = ax[0].contourf(lon, img.r.to('solRad'), np.log10(img.rho.value), levels=levels, cmap=mymap)

# ax[0].plot(0, 0, 'yo', markersize=10, label='Sun')

# ax[0].plot(img.position.longitude.value, img.position.r.to('solRad'), 's', color='magenta', markersize=10, label='STEREO-A')
# coords = fov_coords(img.position.r.to('solRad').value, img.position.longitude.value, 4.0)
# ax[0].plot(coords[0], coords[1], '--', color='magenta')
# coords = fov_coords(img.position.r.to('solRad').value, img.position.longitude.value, 24.0)
# ax[0].plot(coords[0], coords[1], '--', color='magenta')

# ax[0].plot(0, 215, 'o', color='cyan', markersize=10, label='Earth')
# ax[0].patch.set_facecolor('slategrey')
# ax[0].set_xticklabels([])
# ax[0].set_yticklabels([])
# #ax[0].set_xlim(-1.0, 1.0)        
# ax[0].set_ylim(0, 230)
# ax[0].legend(framealpha=1.0)

fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, polar=True)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax = [ax1, ax2, ax3]
levels = np.arange(6.0, 10.0, 0.1)

fig.subplots_adjust(left=0.01, bottom=0.15, right=0.99, top=0.99, wspace=0.15)

pos = ax[0].get_position()
dw = 0.005
dh = 0.05
left = pos.x0 + dw
bottom = pos.y0 - dh
wid = pos.width - 2*dw
cbaxes = fig.add_axes([left, bottom, wid, 0.03])
#cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
#cbar1.ax.set_xlabel("Log[Electron density ($m^{-3}$)]")

# Add label
#time = np.arange(0*24*60*60, 27.0*24*60*60, 40*60)
# Find frame index to save data
label = "Time [days]"
ax[0].text(0.675, -0.0, label, fontsize=14, transform=ax[0].transAxes)

elon = img.elon.to('deg').value
fov = (elon >= 10.0) & (elon <= 24.0)
time = 0 + np.arange(1, jmap.shape[1],1)*240*60/86400

dj = jmap[fov, 1:] - jmap[fov, 0:-1]
ax[1].pcolor(time, elon[fov], dj,vmin=-1e-24,vmax=1e-24,cmap='gray')
ax[1].text(0.05, 0.9, 'Simulation', fontsize=16, transform=ax[1].transAxes, color='navy')

