import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

def fourplot(well, start, end, DEPTH=None, GR=None, DT=None, DEN=None):
    
    if DEPTH is None:
        DEPTH = well.data['DEPT']
    if GR is None:
        GR = well.data['GR']
    if DT is None:
        DT = well.data['AC']
    if DEN is None:
        DEN = well.data['DEN']
    
    gridinc = (end-start)//10
    
    fig, axes = plt.subplots(ncols=4,nrows=1, figsize=(16, 12))

    # Gamma
    axes[0].plot(GR,DEPTH,'k-', alpha=0.5)

    axes[0].set_title('Gamma Ray Log')
    axes[0].set_ylabel('measured depth [m]', fontsize = '12' )
    axes[0].set_xlabel('API', fontsize = '12')

    # Sonic
    axes[1].plot(DT,DEPTH,'k-', alpha=0.5)

    axes[1].set_title('Sonic Log')
    axes[1].set_xlabel('P-wave slowness [usec/m]', fontsize = '12')

    # Density
    axes[2].plot(DEN,DEPTH,'k-', alpha=0.5)

    axes[2].set_title('Density Log')
    axes[2].set_xlabel('Density [kg/m3]', fontsize = '12')

    # Impedance
    axes[3].plot(1e6*DEN/DT,DEPTH,'k-', alpha=0.5)

    axes[3].set_title('Impedance Log (calculated)')
    axes[3].set_xlabel('Impedance [kg/(s*m2)]', fontsize = '12')

    
    # Format the axes
    for i in range(4):
        axes[i].set_ylim((start//gridinc*gridinc),(end//gridinc*gridinc))
        axes[i].set_yticks(np.arange((start//gridinc*gridinc),((end//gridinc+2)*gridinc),gridinc))
        axes[i].invert_yaxis()
        axes[i].grid()

    plt.show()
    
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

def LEvalue(depthlog, log, start, end, adjustment):
    
    """
    
    Log Edit Log Values
    
    returns logout
    
    depthlog - the depth log - assumed to be an MD log, but may work for TVD?
    log - the log which is to be modified
    start - the top of the interval which is to be modified IN DEPTH
    end - the end of the interval which is to be modified IN DEPTH
    adjustment - quanitfies the amount of value change to be imposed
    """
    
    # Find the indecies of the samples that define the top and bottom of adjustment interval:
    start_i = find_nearest(depthlog,start)
    end_i = find_nearest(depthlog,end)
    
    
    # Create adjusted log for new interval
    logout = np.ones_like(log)
    logout[:] = log[:]
    logout[start_i:end_i] = logout[start_i:end_i] + adjustment*np.ones_like(logout[start_i:end_i])
      
    return logout

def LEremove(depthlog, log, start, end):
    
    """
    
    Log Edit Removal
    
    returns logout
    
    depthlog - the depth log - assumed to be an MD log, but may work for TVD?
    log - the log which is to be modified
    start - the top of the interval which is to be modified IN DEPTH
    end - the end of the interval which is to be modified IN DEPTH
    """
    
    

    # Find the indecies of the samples that define the top and bottom of adjustment interval:
    start_i = find_nearest(depthlog,start)
    end_i = find_nearest(depthlog,end)
        
    if start_i-end_i < 0:

        # Create log with interval removed
        logout = np.ones_like(log)
        logout[:] = log[:]
        logout[start_i:-(end_i-start_i)] = logout[end_i:]
        logout[-(end_i-start_i):] = np.nan # erase the portion below


        return logout
    
    else:
        return log
    
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

def LEtimedepth(log,timedepthlog,timelog,inputdepthlog,depthsamplerate):
    """
    
    Log Edit Time to Depth
    
    returns depthlog, logdepth
    
    log - the log which is to be converted to time
    timedepthlog - the log of cumulative time at regular depths
    depthsamplerate - the desired sample rate of the new log (in meters)
    """
    
    # Convert the timedepthlog into a depthtimelog
    # Find the max time, rounded up to the next 0.25 seconds
    #tmax = (np.amax(timedepthlog)//0.25+1)*0.25
    #timelog = np.arange(0, tmax, timesamplerate)
    
    # Calculate the depthtimelog
    depthtimelog = np.interp(timelog,timedepthlog,inputdepthlog)
    
    # Create the depth log, rounding the max depth to the nearest ten meters
    dmax = (np.amax(inputdepthlog)//10+1)*10
    depthlog = np.arange(0, dmax, depthsamplerate)
    
    # Interpolate the log
    logdepth = np.interp(depthlog,depthtimelog,log)
    
    return depthlog, logdepth

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

