## Get started - Set up this workspace:
    
import numpy as np
import matplotlib.pyplot as plt

import bruges # pip install bruges     installs a great library from Agile Scientific

from skimage.filters import gaussian
from skimage.util import random_noise

from ipywidgets import FloatSlider, IntSlider, fixed, FloatLogSlider, interactive #interact, interact_manual, IntText,
import ipywidgets as widgets
from IPython.display import display
import matplotlib.patches as patches
from statistics import pvariance

# Functions from LEtools.py

def find_nearest(array, value):
    # find the index of nearest sample to a given value
    idx = (np.abs(array - value)).argmin()
    return idx

def LEthink(depthlog, log, start, end, adjustment):
    
    """
    
    Log Edit Thin/Thick
    
    returns depthout, logout
    
    depthlog - the depth log - assumed to be an MD log, but may work for TVD?
    log - the log which is to be modified
    start - the top of the interval which is to be modified IN DEPTH
    end - the end of the interval which is to be modified IN DEPTH
    adjustment - quanitfies the amount of thickness change to be imposed (positive is thickening)
    """
    
    if adjustment != 0:
        
        # Find the indecies of the samples that define the top and bottom of adjustment interval:
        start_i = find_nearest(depthlog,start)
        end_i = find_nearest(depthlog,end)


        # Create depthlog for new interval

        # Find the depthlog sample rate
        deltad = depthlog[1]-depthlog[0]

        # Create new depthlog for interval
        depthinterval = np.arange(start,end+adjustment,deltad)


        # Create adjusted log for new interval

        # Extract the portion of the log to be adjusted
        loginterval = log[start_i:end_i]

        # Interpolate/resample
        #logintersamp = np.interp(np.arange(start,end,(end-start)/(end_i-start_i+(adjustment/deltad))),depthlog[start_i:end_i],loginterval)
        logintersamp = np.interp(np.arange(start,end,(end-start)/len(depthinterval)),depthlog[start_i:end_i],loginterval)

        # Ensure length of output logs is the same... squeezing logs sometimes caused a problem
        if len(logintersamp)!= len(depthinterval):
            logintersamp = logintersamp[:-1]


        # Create new depthlog for output
        depthout = np.append(np.append(depthlog[:start_i],depthinterval),depthlog[end_i:]+adjustment*np.ones_like(depthlog[end_i:]))

        # Create new log for output
        logout = np.append(np.append(log[:start_i],logintersamp),log[end_i:])

        return depthout,logout
    
    else:
        return depthlog,log

def LEdepthtime(log,timedepthlog,timesamplerate):
    """
    
    Log Edit Depth to Time
    
    returns timelog, logout
    
    log - the log which is to be converted to time
    timedepthlog - the log of cumulative time at regular depths
    timesamplerate - the desired sample rate of the new log (in seconds, often 0.002)
    """
    
    tmax = (np.amax(timedepthlog)//0.25+1)*0.25 # rounds to the nearest 0.25 seconds
    
    # Create the timelog
    timelog = np.arange(0, tmax, timesamplerate) # Creates the log into which values are interpolated
    
    # Interpolate the log
    logtime = np.interp(timelog,timedepthlog,log)
    
    return timelog, logtime

def logs2RC_t(depthlog,density,sonic,timesamplerate,depthsamplerate=0.2,vreplacement=2000):
    '''
    For now, only input density and sonic logs. Other logs may be added later!
    
    returns a time log, sonic log, and a reflectivity series
    
    depthlog : the depth log
    density : the density log, used to calculate the impedance
    sonic : the sonic log, used to calculate the impedance
    timesamplerate : the desired samplerate for the output (in seconds, often 0.002)
    depthsamplerate : the samplerate of the logs from the LAS file (in meters, often 0.2)
    vreplacement : the assumed velocity above the top of the sonic log (meters/second)
    
    
    '''
    # Calculate the time-depth relationship
    # Convert DT to travel time per sample, then create a running total
    cumtime = 2 * np.cumsum(depthsamplerate/1e6 * np.nan_to_num(sonic))
    # Account for the time above the shallowest measurement
        # Find shallowest non-zero sonic measurement, convert that depth to time via
        # replacement velocity, then add that value to cumtime values.
    timedepth = (2 * depthlog[np.amin(np.where(cumtime[:] != 0))]/vreplacement) + cumtime
    
    # Convert logs from depth to time
    timelog, density_t = LEdepthtime(density,timedepth,timesamplerate)
    timelog, sonic_t = LEdepthtime(sonic,timedepth,timesamplerate)
    
    # Calculate impedance log
    impedance_t = 1e6*(density_t/sonic_t)
    
    # Calculate the reflectivity series
    reflectivity_t = (impedance_t[1:] - impedance_t[:-1]) / (impedance_t[1:] + impedance_t[:-1])
    
    return timelog[:-1], sonic_t, reflectivity_t


# Get started - Create synthetic sonic and density logs for a well.

# Density Log
DEN = np.zeros(500)
DEN[:250] = 2300
DEN[250:] = 2500

# Sonic Log
DT = np.zeros_like(DEN)
DT[:250] = 375 # ~2650 m/s
DT[250:] = 350 # ~2850 m/s

# Depth track
DEPTH = np.array(np.arange(0,100,0.2))


def fault_model(freq,f_offset,noise_var,lat_smooth,DEPTH,DEN,DT,dt=0.001,len_wav=0.2):
    """
    Function to demonstrate the visibility of faults. Generates a plot.
    
       
    #corner_freqs  =  list of four corner frequencies for Ormsby wavelet [f1,f2,f3,f4]
    freq          =  central frequency of Ricker wavelet
    f_offset      =  vertical offset of the modelled fault
    noise_var     =  variance of random noise (= (standard deviation)**2). Best >> 1
    lat_smooth    =  lateral smoothing defined as standard deviation for gaussian kernal.
    DEPTH         =  Depth log
    DEN           =  Density log
    DT            =  Sonic log
    dt            =  sample rate in seconds for modelling and wavelet (default=0.001)
    len_wav       =  length in seconds for wavelet (default = 0.2)
    
    """
    
    # Define the wavelet

    wav = bruges.filters.wavelets.ricker(duration=len_wav,dt=dt,f=freq)
    wavelength = np.sqrt(6)/(np.pi*freq)* 1e6 /DT[0] /2

    # Create a new time series for the wavelet - may not use it, but it's here
    Twavestart = -(len_wav/dt-1)/2 * dt
    Twaveend = (len_wav/dt-1)/2 * dt

    Twav = np.arange(Twavestart,Twaveend+dt,dt)

    # Use a Bag of Really Useful Geophysical Stuff to calculate the spectrum of the un-interpolated wavelet:
    spec = bruges.attribute.spectrogram(wav,len(wav)-1,zero_padding=100)[0]

    # Calculate the Nyquist frequency from the sample rate:
    nyq = 0.5 * (1/dt) # samplerate is in seconds
    
    
    # Run TIME
    
    # Run this to create a time log. We're not interested in the time log, per se,
    # but it is required to run the flow
    TIME,DT_t,RC_t = logs2RC_t(DEPTH,DEN,DT,dt)
    
    
    # Create model

    # Initialize an array to hold the wedge:
    secRC = np.array([[]])
    secDT = np.array([[]])
    secDT_time = np.array([[]])

    # Define left side of model
    for i in range(0,40):

        # Modify logs - not changing these traces    
        # Thicken the DT log between 0m and 10m by 0m
        depth_interim, DT_interim = LEthink(DEPTH,DT,0,10,0)
        depth_interim, DEN_interim = LEthink(DEPTH,DEN,0,10,0)

        # Calculate reflectivity in time
        # Use Log Edit tools to calculate Reflectivity Coefficients and convert trace to time
        time_interim, DT_time, RC_interim = logs2RC_t(depth_interim,
                                                              DEN_interim,
                                                              DT_interim,dt)

        # Append to section as new trace
        if i == 0: # if-else is used to deal with the case array is empty
            secRC = np.append(secRC,[RC_interim[:len(TIME)]],axis=1)
            secDT = np.append(secDT,[DT_interim[:len(DEPTH)]],axis=1)
            secDT_time = np.append(secDT_time,[DT_time[:len(TIME)]],axis=1)
        else:
            secRC = np.append(secRC,[RC_interim[:len(TIME)]],axis=0)
            secDT = np.append(secDT,[DT_interim[:len(DEPTH)]],axis=0)
            secDT_time = np.append(secDT_time,[DT_time[:len(TIME)]],axis=0)


    # Define right side of model
    for i in range(0,40):

        # Modify logs - add extra thickness to top of log    
        # Thicken the DT log between 0m and 10m by fault offset
        depth_interim, DT_interim = LEthink(DEPTH,DT,0,10,f_offset)
        depth_interim, DEN_interim = LEthink(DEPTH,DEN,0,10,f_offset)

        # Calculate reflectivity in time
        # Use Log Edit tools to calculate Reflectivity Coefficients and convert trace to time
        time_interim, DT_time, RC_interim = logs2RC_t(depth_interim,
                                                              DEN_interim,
                                                              DT_interim,dt)

        # Append to section as new trace
        secRC = np.append(secRC,[RC_interim[:len(TIME)]],axis=0)
        secDT = np.append(secDT,[DT_interim[:len(DEPTH)]],axis=0)
        secDT_time = np.append(secDT_time,[DT_time[:len(TIME)]],axis=0)

        
    # Add noise - I found adding noise before convolution with the wavelet
    #             created more realistic results.
    
    # Add noise to RC model:
    secRC_noise = random_noise(secRC, mode='gaussian', var=noise_var)

       
    # Convolve with wavelet
    
    # Initialize empty array
    sec_fault_synth = np.zeros_like(secRC_noise)
    # Populate array with reflectivity amplitude
    for i in range(secRC.shape[0]):
        sec_fault_synth[i,:] = np.convolve(wav, np.nan_to_num(secRC_noise[i,:]), mode='same')

    # Extract the max amplitude from the synthetic - used for scaling plots
    synthAMPmax = 1.0 * max(np.amax(sec_fault_synth),abs(np.amin(sec_fault_synth)))
    
    
    # Lateral smoothing - gaussian smoothing applied across each time sample
    
    sec_fault_synth_filt = gaussian(sec_fault_synth,
                                      sigma=[lat_smooth,0], mode='reflect',
                                      preserve_range=True)
    
    
    # Calculate signal to noise ratio
    
    #SNR = Variance(signal) / (Variance(signal + noise) - Variance(signal))
    # Need Variance of signal, and coveniently the variance of one trace
    # is same as whole section
    var_signal = pvariance(np.convolve(wav, np.nan_to_num(secRC[0]), mode='same'))
    
    # Now calculate the ratio                       
    snr = var_signal / (pvariance(sec_fault_synth_filt.tolist()[0])- var_signal)
    
    
    # Plot spectrogram, model in depth, model in time, final synthetic
    
    #Plot Wavelet
    #fig = plt.figure(figsize=(12,12))
    fig, ax = plt.subplots(1, 2, figsize=(10,2))
    ax[0].plot(Twav,wav,c='k',linewidth=3, alpha=0.5)
    ax[0].axhline(y=0, c='k')
    ax[0].set_title('Ricker Wavelet')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Amplitude')
    ax[0].grid()

    # plot the histogram
    ax[1].plot(np.arange(0,nyq,nyq/len(spec)),spec, c='k',linewidth=3, alpha=0.5)
    ax[1].set_yscale('log')
    ax[1].set_xlim(0,3*freq)
    ax[1].set_ylim(bottom = 0.002)
    ax[1].set_title('Wavelet Frequency Spectrum')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Amplitude')
    ax[1].grid()

    # Second half of composite is created as new figure!
    # Sonic in time
    fig2, ax = plt.subplots(1, 2, sharey=True, figsize=(10,6))
    im = ax[0].imshow(secDT_time[:,:75].T, vmin=340, vmax=410,
                   cmap="viridis_r", aspect='auto')
    ax[0].yaxis.set_ticks_position("both")
    #ax[0].yaxis.set_label_position("right")
    ax[0].set_yticks(range(0,71,10))
    ax[0].set_yticklabels(np.arange(0,71*dt, 10*dt))
    ax[0].set_title('Section: Slowness')
    ax[0].set_ylabel('Time [ms]')
    ax[0].set_xlabel('trace')
    #ax[0].axvline(x=40, linestyle='--', color='k')
    cbar = fig.colorbar(im, ax=ax[0],ticks=range(200,401,50),orientation='horizontal')
    cbar.ax.invert_xaxis()
    cbar.set_label('slowness [usec/m]')
    
    # Add some text:
    ax[1].text(-40,-20,'Fault offset is ' + r'$\lambda/ %s$' % np.round(wavelength/f_offset,1), fontsize=15)
    ax[1].text(-40,-10,'Signal to noise ratio is %s' % np.round(snr,2), fontsize=15)

    # Wavelet seismic in time
    #ax = fig.add_subplot(236)
    im = ax[1].imshow(sec_fault_synth_filt[:,:75].T,
                   vmin=-synthAMPmax, vmax=synthAMPmax,
                   cmap="seismic", aspect='auto')
    ax[1].set_yticks(range(0,71,10))
    ax[1].set_yticklabels(np.arange(0,71*dt, 10*dt))
    ax[1].set_title('Section: Reflectivity')
    #ax[1].set_ylabel('Time [ms]')
    ax[1].set_xlabel('trace')
    #ax[1].axvline(x=40, linestyle='--', color='k')
    cbar = fig.colorbar(im, ax=ax[1], ticks=[-.1,0,.1], orientation='horizontal')
    cbar.set_label('reflectivity')


rslt = interactive(fault_model, freq=IntSlider(min=10, max=150, step=2, description='Frequency', value=60, continuous_update=False),
                   f_offset=IntSlider(min=0, max=20, step=1, description='Fault Offset', value = 10, continuous_update=False),
                   noise_var=FloatLogSlider(value=0.001, base=10, min=-6, max=-1, description='Noise Var.', continuous_update=False),
                   lat_smooth=IntSlider(min=0, max=10, step=1, value=3, description='Lat. Smooth', continuous_update=False),
                   DEPTH=fixed(DEPTH), DEN=fixed(DEN), DT=fixed(DT),
                   dt=fixed(0.001),len_wav=fixed(0.2))



display(rslt)

