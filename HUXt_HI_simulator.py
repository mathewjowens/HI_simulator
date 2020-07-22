# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:04:08 2020

@author: mathewjowens
"""
import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import HI_simulator as HI

#change to the HUXt dir so that the config.dat is found
os.chdir(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))
import HUXt as H

fig_dir = "D:\\Dropbox\\python_repos\\HI_simulator\\Figures"


# <codecell>  Run HUXt

cr=2210
vr_in, br_in = H.Hin.get_MAS_long_profile(cr)

#create a density map assuming constant mass flux
n_in=vr_in.value*0.0 + 4.0
n_in=n_in + ((650 - vr_in.value) /400)*5.0

#now run HUXt - use dt_scale =5, as that's around a 40-min timestep.
model = H.HUXt(v_boundary=vr_in, cr_num=cr, br_boundary=br_in, rho_boundary=n_in,
               latitude=0*u.deg, 
               lon_start=(3*np.pi/2)* u.rad, lon_stop=(np.pi/2)* u.rad,
               simtime=3*u.day, dt_scale=5)
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, latitude=0.0*u.deg,
                width=60*u.deg, v=1500*(u.km/u.s), thickness=5*u.solRad)
model.solve([cme]) 

t = 1.5*u.day
#model.plot_radial(t, lon=200.0*u.deg,field='br_ambient')
#model.plot_radial(t, lon=200.0*u.deg,field='ambient')
model.plot(t, field='v')
model.plot(t, field='br')

model.rho_post_process()
model.plot(t, field='rho')

# <codecell>  Produce the Jamp
HI.Jmap_HUXt(model, obs= Observer(0.976, 45.0))
HI.Jmap_HUXt(model, obs= Observer(0.976, 90.0))

# <codecell> Thomson scattering diagnostic frames (also generates Jmap)

#plot a single time step
t=20
#obs = Observer(0.976, 47.0) # STA, R(AU), long(deg)
obs = HI.Observer(0.976, 90.0) # STA, R(AU), long(deg)
img = HI.Imager(obs, np.flipud(np.rot90(model.rho_grid[t,:,:])), model.r, model.lon, model.latitude)

#find the equatorial plane and plot
fig,ax=plt.subplots(figsize=(10,10))
p=ax.scatter(img.grid_x,img.grid_y,c=img.modelrho)
plt.scatter(img.x,img.y,c=img.rho)


time = model.time_out
#generate movie of equatorial plane, the electron density along LOS and brightness along LOS
for i, t in enumerate(time):
    
    # #rotate the solution
    # Tsid=2192832
    # spinlong=rho_long + 2*np.pi *u.rad * t / Tsid
    rho=np.flipud(np.rot90(model.rho_grid[i,:,:]))
    
    img = HI.Imager(obs, rho, model.r, model.lon, model.latitude)
    HI.plot_frame(i, t, sun, obs, img, savefig=True, tag="HI_frame",fig_dir=fig_dir)
    if i == 0:
        jmap = np.zeros((img.I.size, time.size))
        
    jmap[:, i] = img.I.value
    

    
#animate
###############################################################################
out_name = "HI_frame_t*.png"
src = os.path.join(fig_dir, out_name)
gif_name = "HI_HUXt.gif"
dst = os.path.join(fig_dir, gif_name)
HI.make_animation(src, dst, tidy=True)



#plot the Jmap
###############################################################################
mymap = mpl.cm.cividis
mymap.set_over([1, 1, 1])
mymap.set_under([0, 0, 0])
dv = 10


fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax = [ax1, ax2]
levels = np.arange(6.0, 10.0, 0.1)

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

