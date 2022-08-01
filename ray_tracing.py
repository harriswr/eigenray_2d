#!/home/harriswr/codes/anaconda3/bin/python

from audioop import mul
from math import pi, ceil
from re import I
import numpy as np
from os import listdir
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from copy import deepcopy, copy
import pyautogui
import multiprocessing
from joblib import Parallel, delayed
import time

class RayTrace():
    '''Remake of ray trace that should be cleaner and easier to troubleshoot'''

    def __init__(self):
        '''Initialize variables so that consistent dictionary definitions 
        between all test cases'''
        self.bathy_defined = False
        self.ssp_defined = False
        self.defined_slowness_fxn = False
        self.niw_waveform = False

        self.wd={} #water depth
        self.ssp={} #sound speed profile
        self.rays=[] #container for holding ray definitions

    def _binterp_slowness(self, x, z):
        ''' Use bilinear interpolation to determine slowness profile at the
        designated (x,z) point.

        Parameters:
        ------------
        x: Array-like. X-location to interpolate to in meters relative 
            to (0,0).
        z: Array-like. Z-location to interpolate to in meters relative 
            to (0,0).

        Outputs:
        -----------
        temp_slw: Interpolated value of the slowness field in sec per meter
        temp_dslwdx: Interpolated value of the slope of the slowness field
            with respect to x in sec per meter per meter
        temp_dslwdz: Interpolated value of the slope of the slowness field
            with respect to z in sec per meter per meter
        '''

        if not hasattr(x, '__iter__'):
            x = np.array(x, ndmin=1)

        if not hasattr(z, '__iter__'):
            z = np.array(z, ndmin=1)
            
        nx, nz = self.slowness['SSP'].shape
        
        temp_slw = np.zeros_like(x, dtype=np.float64)
        temp_dslwdx = np.zeros_like(x, dtype=np.float64)
        temp_dslwdz = np.zeros_like(x, dtype=np.float64)

        if nx > 1:
            x[x<self.wd['x0']] = self.wd['x0']
            x[x>=self.wd['x1']] = self.wd['x1'] - 1.e-6
            x = (x-self.wd['x0']) / self.wd['dx']
            i = np.floor(x).astype(int) + 1
            iplusone = i + 1
            x = x - i + 1
        else:
            i = np.ones_like(x).astype(int)
            iplusone = np.ones_like(x).astype(int)
            x = np.zeros_like(x)

        if nz > 1:
            z0 = np.min(self.ssp['SSP_Z'])
            z1 = np.max(self.ssp['SSP_Z'])
            dz = self.ssp['SSP_Z'][1]-self.ssp['SSP_Z'][0]
            z[z<z0] = z0
            z[z>=z1] = z1 - 1.e-6
            z = (z-z0)/dz
            k = np.floor(z).astype(int) + 1
            kplusone = k + 1
            z = z - k + 1
        else:
            k = np.ones_like(z).astype(int)
            kplusone = np.ones_like(z).astype(int)
            z = np.zeros_like(z)
        
        #subtract one from indexing since python is 0 index base
        #flip this tag to use flattened array versus index lookup
        p00 = self.slowness['SSP'][i-1,k-1]
        p01 = self.slowness['SSP'][i-1,kplusone-1]
        p10 = self.slowness['SSP'][iplusone-1,k-1]
        p11 = self.slowness['SSP'][iplusone-1,kplusone-1]

        B1 = np.array([[1, 0, 0, 0], [-1, 0, 1, 0], [-1, 1, 0, 0],\
             [1, -1, -1, 1]])

        if not hasattr(p00, '__iter__'):
            p00 = np.asarray([p00])
            p01 = np.asarray([p01])
            p10 = np.asarray([p10])
            p11 = np.asarray([p11])

        for ai, _ in enumerate(x):
            temp_term = np.array([p00[ai], p01[ai], p10[ai], p11[ai]])@B1.T
            temp_slw[ai] = temp_term@np.array([1, x[ai], z[ai], x[ai]*z[ai]]).T
            temp_dslwdx[ai] = temp_term@np.array([0, 1, 0, z[ai]]).T
            temp_dslwdz[ai] = temp_term@np.array([0, 0, 1, x[ai]]).T

        if nx > 1:
            temp_dslwdx = temp_dslwdx / self.wd['dx']
        if nz > 1:
            temp_dslwdz = temp_dslwdz / dz
        
        return temp_slw, temp_dslwdx, temp_dslwdz
        
    def _binterp_depth(self, x):
        ''' Use bilinear interpolation to determine water depth at the
        designated x point.

        Parameters:
        ---------------
        x: Array-like. X-location to interpolate to in meters relative to 
            (0,0).

        Outputs:
        -----------
        wd: Interpolated value of the seafloor depth at the specified
            x-location in meters.
        normal: Array-like. Normal vector of the seafloor at the specified 
        x-location.
        '''
        if not hasattr(x, '__iter__'):
            x = np.asarray([x])

        wd = np.empty_like(x)
        dwdx = np.empty_like(x)
        nx = len(self.wd['field'])

        if nx > 1:
            x[x<self.wd['x0']] = self.wd['x0']
            x[x>=self.wd['x1']] = self.wd['x1']-1.e-6
            x = (x-self.wd['x0'])/self.wd['dx']
            i = np.floor(x).astype(int) + 1
            iplusone = i + 1
            x = x - i + 1
        else:
            i = np.ones_like(x).astype(int)
            iplusone = np.ones_like(x).astype(int)
            x = np.zeros_like(x)

        #subtract one from indexing since python is 0 index base
        p0 = self.wd['field'][int(i-1)]
        p1 = self.wd['field'][int(iplusone-1)]

        B1 = np.array([[1,0], [-1,1]])

        if not hasattr(p0, '__iter__'):
            p0 = np.asarray([p0])
            p1 = np.asarray([p1])

        for ai, _ in enumerate(x):
            _temp = np.array([p0[ai], p1[ai]])@B1.T
            wd[ai] = _temp@np.array([1, x[ai]]).T
            dwdx[ai] = _temp@np.array([0,1]).T

        if nx > 1:
            dwdx = dwdx/self.wd['dx']
        dwdz = -1*np.ones_like(x)
        norm = np.sqrt(dwdx**2+dwdz**2)

        dwdx = np.reshape(dwdx/norm, x.shape)
        dwdz = np.reshape(dwdz/norm, x.shape)

        return wd, np.array([dwdx, dwdz]).squeeze().T
    
    def _eikonal(self, _, y, ref_pt):
        '''
        Implementation of the eikonal equation for use in ODE solve.

        Parameters:
        -------------
        _: Unused variable that is antiquated from Matlab code.
        y: Vector of variables to be solved
            y = [x, dxi/ds, z, dzeta/ds, tau]
        ref_pt: Current reference point in the ray tracing loop. This should be
            formatted as a dictionary with keys for x, z, tau, and s

        Outputs:
        ------------
        y_dot: vector of sol'ns to the eikonal ode
            y_dot = [dxi/ds * 1/c, dslownessdx, dzeta/ds * 1/c, 
                    dslownessdz, slowness]
        '''
        y = np.array(y).squeeze()
        var1 = y[0] + ref_pt['x']
        var2 = y[2] + ref_pt['z']
        slw, dslwdx, dslwdz = self._binterp_slowness(var1, var2)
        y_dot = np.array([y[1]/slw, dslwdx, y[3]/slw, dslwdz, slw]).T
        return y_dot

    def _get_xfan(self, theta):
        '''
        Function to return the x and z traces of a ray provided an initial
        angle theta. 
        
        This is not meant to be a standalone function but is included in order 
        to be parallelized for loops.
        
        Parameters
        ------------
        theta: Initial ray take-off angle relative to horizontal in degrees.
        
        Outputs:
        ------------
        ray_x: History of x-coordinates for ray at each step
        ray_z: History of z-coordinates for ray at each step
        '''
        ray_x, ray_z, *_ = self.create_ray_trace(theta)
        return ray_x, ray_z
    
    def _refine_angle(self, i, tol=1e-6):
        '''
        Brute force zero-crossing method to determine eigenangles. Angle is
        determined as when the difference between the receiver z-location and
        ray termination z-location is less than the specified tolerance.

        This is not meant to be a standalone function but is included in order 
        to be parallelized for loops.

        Parameters:
        --------------
        i: Index of eigenangle array to be refined
        tol: Maximimum difference between receiver z and ray termination z 
            before refinement loop can end.

        Outputs:
        -------------
        th0: Angle on one side of zero crossing in degrees.
        th1: Angle on other side of zero crossing in degrees
        z_bi0: Z-location of ray terminal point for th0 in meters.
        z_bi1: Z-location of ray terminal point for th1 in meters.
        ray_tau: Travel time in seconds of final eigenray from source to 
            receiver.
        '''
        z_bi0 = self.zero_crossing0[i]
        z_bi1 = self.zero_crossing1[i]
        th0 = self.angle0[i]
        th1 = self.angle1[i]
        print(f'Refine eigenangle between {th0:.2f} deg and {th1:.2f} deg')

        while abs(th1-th0) > tol:
            thnew = (th0 + th1)/2.
            _, ray_z, ray_tau, *__ = self.create_ray_trace(thnew)
            z_binew = ray_z[-1]
            z_binew = z_binew - self.rcvr['z']
            if z_bi0*z_binew <= 0:
                th1 = thnew
                z_bi1 = z_binew
            else:
                th0 = thnew
                z_bi0 - z_binew
        
        return [th0, th1], [z_bi0, z_bi1], ray_tau

    def _seasurf_level(self, x):
        '''
        Returns sea surface height as an array of zeros with size like x

        Parameters:
        -------------
        x: Array of x-locations for which ray-tracing algorithm is to be run.

        Outputs:
        ------------
        ssl: Sea surface level in m. Assumed to be zero in all cases.
        norm: Sea surface normal. Assumed to be [0,1] in all cases.
        '''    
        if type(x) != np.ndarray:
            x = np.array(x)

        norm = np.array([0,1]).T
        ssl = np.zeros_like(x)
        return ssl, norm

    def create_ray_trace(self, theta, abstol=1e-9, reltol=1e-6, ds=10, \
        nmaxbtmbnce=10, ode_method='RK45', use_max_ds=True):
        ''' Creates a history of ray trace provided that bathymetry and
        sound speed profile have been defined. 
        
        Ray tracing is accomplished by numerical integration using SciPy's 
        solve_ivp method with events specified for:

                1. surface reflection
                2. bottom reflection
                3. reaching domain boundary
                4. water depth is zero
                5. (optional) reach maximum trace time

        Parameters:
        ------------
        theta: Initial ray take-off angle in degrees.
        abstol: Absolute tolerance for numerical error in the ODE solver.
            Default value is 1.e-9.
        reltol: Relative tolerance for numerical error in the ODE solver.
            Default value is 1.e-6.
        ds: Maximum step size allowed for the Runge-Kutta marching algorithm.
            Measured in meters along the arclength of the resultant ray.
            Default value is 10.
        nmaxbtmbnce: Maximum number of bottom bounces before ray is terminated.
            Default value is 10.
        ode_method: Specifies solve_ivp ODE marching method. Valid inputs are
            RK23, RK45, DOP853, and LSODA (in order of increasing order and
            accuracy). Default value is RK45.
        use_max_ds: Flag to disable the use of maximum ds step in solve_ivp.
            In order to most closely, match Matlab results and minimize
            numerical errors, this should be set to True with a ds as stated
            above.

        Outputs:
        -----------
        x: Time history of x-location for each evaluation point in solve_ivp
        z: Time history of z-location for each evaluation point in solve_ivp
        tau: Time history of ray travel time for each evaluation point in 
            solve_ivp
        s: Time history of ray arclength for each evaluation point in solve_ivp
        tang: Time history of ray tangent angle for each evaluation point in 
            solve_ivp
        reflections: Information about each reflection that occurred during the
            numerical intergration of the ray. Information provided includes:

                [Arclength at time of reflection,
                    X-location of reflection,
                    Z-location of reflection,
                    Incident ray angle prior to reflection,
                    Normal of reflecting surface]

        slw: Time history of slowness field for each evaluation point in 
            solve_ivp
        dslwdx: Time history of slope of slowness field relative to x for each 
            evaluation point in solve_ivp
        dslwdz: Time history of slope of slowness field relative to z for each 
            evaluation point in solve_ivp
        '''

        if (self.bathy_defined == False or self.ssp_defined == False):
            print('Must specify both bathymetry and sound speed profile\
                prior to attempting ray trace.')
            return

        slw0, *_ = self._binterp_slowness(self.source['x'],self.source['z'])
        dxds0 = np.cos(theta/180.*pi)
        dzds0 = np.sin(theta/180.*pi)
        xi0 = float(dxds0 * slw0)
        zeta0 = float(dzds0 * slw0)
        
        #initialize empty arrays
        x = np.array([]) #x-pos
        z = np.array([]) #z-pos
        tau = np.array([]) #travel time
        s = np.array([]) #arc length
        tang = np.array([]) #ray tangent angle
        self.reflections = []
        ibnce = 0 #number of bottom bounces
        isterm = 0 #a flag to terminate ray tracing routing

        #initial reference point is at the source with no travel arc or time
        ref_pt = {}
        ref_pt['x'] = copy(self.source['x'])
        ref_pt['z'] = copy(self.source['z'])
        ref_pt['tau'] = 0
        ref_pt['s'] = 0

        #term. ray trace for following conditions:
        #   1. surface reflection
        #   2. bottom reflection
        #   3. reaching domain boundary
        #   4. water depth is zero
        #   5. (optional) reach maximum trace time

        def _surface_reflection(s, y, ref_pt=ref_pt):
            sea_surface_level, _ = self._seasurf_level(y[0] + ref_pt['x'])
            output = y[2]+ref_pt['z']-sea_surface_level
            return output

        def _bottom_reflection(s, y, ref_pt=ref_pt):
            sea_floor_depth, _ = self._binterp_depth(y[0] + ref_pt['x'])
            output = sea_floor_depth-(y[2]+ref_pt['z'])
            return output

        def _reach_bounds(s, y, ref_pt=ref_pt):
            dist_to_bndry = np.min([y[0]+ref_pt['x']-self.wd['x0'],\
                self.wd['x1']-(y[0]+ref_pt['x'])])
            return dist_to_bndry

        def _zero_depth(s, y, ref_pt=ref_pt):
            sea_floor_depth, _ = self._binterp_depth(y[0] + ref_pt['z'])
            return sea_floor_depth

        def _max_time(s, y, ref_pt=ref_pt, tmax=1e10):
            output = tmax-(y[4]+ref_pt['tau'])
            return output

        #use lambda so that can define as terminal events
        surface_event = lambda s,y:_surface_reflection(s,y)
        surface_event.terminal = True
        surface_event.direction = -1

        bottom_event = lambda s,y:_bottom_reflection(s,y)
        bottom_event.terminal = True
        bottom_event.direction = -1

        boundary_event = lambda s,y: _reach_bounds(s,y)
        boundary_event.terminal = True
        boundary_event.direction = -1

        depth_event = lambda s,y: _zero_depth(s,y)
        depth_event.terminal = True
        depth_event.direction = -1

        time_event = lambda s,y: _max_time(s,y)
        time_event.terminal = True
        time_event.direction = -1

        self.events=[]
        self.yevents=[]

        self.test_var=[]

        #Turn off solve_ivp warnings for list formats
        np.warnings.filterwarnings('ignore', \
            category=np.VisibleDeprecationWarning)

        #initialize variables that will be iterated in the while loop
        x0 = self.source['x']
        z0 = self.source['z']
        tau0 = 0
        s0 = 0

        while 1:
            
            #initial condition for this branch of ray trace
            y0 = np.array([x0 - ref_pt['x'], xi0, z0-ref_pt['z'], zeta0,\
                tau0-ref_pt['tau']]).T
            self.yevents.append(y0)

            span = [s0-ref_pt['s'], np.inf]
            eikonal = lambda s,y: self._eikonal(s, y, ref_pt=ref_pt)
            if use_max_ds:
                sol = solve_ivp(eikonal, t_span=span, y0=y0, atol=abstol,\
                    dense_output=True, method=ode_method, rtol=reltol, \
                    events=(surface_event,bottom_event,boundary_event\
                    ,depth_event,time_event), max_step=ds)
            else:
                sol = solve_ivp(eikonal, t_span=span, y0=y0, atol=abstol,\
                    dense_output=True, method=ode_method, rtol=reltol, \
                    events=(surface_event,bottom_event,boundary_event\
                    ,depth_event,time_event))

            ###
            self.sol = sol
            ###
            event_id = 0
            min_t = 0

            #determine which event caused termination
            for aid, aevent in enumerate(sol.t_events):
                if aevent.size > 0:
                    if aevent[0] > min_t:
                        min_t = aevent[0]
                        event_id = int(aid)

            self.events.append(min_t)

            t_arr = np.arange(0, sol.t[-1]+ds, step=ds)

            #exclude termination point
            t_arr = t_arr[t_arr < sol.t_events[event_id][-1]].T
            y = sol.sol(t_arr).T
            
            #recall: y = [x, dxi/ds, z, dzeta/ds, tau]
            x = np.append(x, y[:,0] + ref_pt['x'])
            z = np.append(z, y[:,2] + ref_pt['z'])
            tau = np.append(tau, y[:,4] + ref_pt['tau'])
            tang = np.append(tang, np.arctan2(y[:,3], y[:,1]))
            s = np.append(s, t_arr+ref_pt['s'])

            #event handling
            event_y = sol.y_events[event_id].squeeze()
            #self.test_var.append(event_y)
        
            if event_id == 0:
                _, normal = self._seasurf_level(event_y[0] + ref_pt['x'])
                ref_data = [sol.t_events[event_id][-1] + ref_pt['s'],\
                    event_y[0]+ref_pt['x'],
                    event_y[2]+ref_pt['z'],
                    np.arctan2(event_y[3], event_y[1]),
                    np.arctan2(normal[1], normal[0])]
                self.reflections.append(ref_data)
            
            elif event_id == 1:
                ibnce += 1
                _, normal = self._binterp_depth(event_y[0] + ref_pt['x'])
                if ibnce >= nmaxbtmbnce:
                    isterm = 1
                ref_data = [sol.t_events[event_id][-1] + ref_pt['s'],\
                    event_y[0]+ref_pt['x'],
                    event_y[2]+ref_pt['z'],
                    np.arctan2(event_y[3], event_y[1]),
                    np.arctan2(normal[1], normal[0])]
                self.reflections.append(ref_data)

            else:
                isterm = 1

            if isterm == 1:
                x = np.append(x, event_y[0]+ref_pt['x'])
                z = np.append(z, event_y[2]+ref_pt['z'])
                tau = np.append(tau, event_y[4]+ref_pt['tau'])
                tang = np.append(tang, np.arctan2(event_y[3], event_y[1]))
                s = np.append(s, sol.t_events[event_id][0]+ref_pt['s'])
                break
                
            x0 = event_y[0] + ref_pt['x']
            xi0 = event_y[1]
            z0 = event_y[2] + ref_pt['z']
            zeta0 = event_y[3]
            tau0 = event_y[4] + ref_pt['tau']
            s0 = sol.t_events[event_id][0] + ref_pt['s']
            
            ei=np.array([xi0, zeta0]).T

            def _reflect(ei, normal):
                return ei - 2*(ei.T@normal)*normal

            er = _reflect(ei, normal)
            self.test_var.append([deepcopy(ref_pt), [x0, xi0, z0, zeta0, tau0,\
                 s0],[ei, normal, er]])
            xi0 = er[0]
            zeta0 = er[1]

            ref_pt['x'] = x0
            ref_pt['z'] = z0
            ref_pt['tau'] = tau0
            ref_pt['s'] = s0

        slw, dslwdx, dslwdz = self._binterp_slowness(x, z)

        return x,z,tau,s,tang,self.reflections,slw,dslwdx,dslwdz

    def define_bathymetry(self, constant_water_depth=None, fname=None):
        '''
        Reads bathymetry file or establishes field with constant depth. 
        If constant_water_depth is provided, additional user input will be
        prompted concerning the x limits and the x-spacing.

        Note that this function does not return outputs but defines class
        attributes critical to the functionality of other attributes.

        Parameters:
        ------------
        constant_Water_depth: Value of constant water depth to be assumed
            across entire domain.
        fname: Filename of .mat file from which to read bathymetry data. 
            Assumes that .mat file has fields of:

            x0: Minimum x-value in domain
            x1: Maximum x-value in domain
            dx: Separation between different subsequent range locations.
            field: Array of water depth in meters.
        
        If a filename is provided, it will take precedence over a constant
        water depth.

        Outputs:
        -----------
        None.
        '''

        if (constant_water_depth != None and fname != None):
            print('Constant depth and file specified. File takes precedence.')
            return

        if fname != None:
            self.wd = loadmat(fname)

            for akey in self.wd.keys():
                try:
                    self.wd[akey] = self.wd[akey].flatten()
                except:
                    print(f'...Bathymetry: {akey} was not flattened.')

            self.bathy_defined = True
        
        elif constant_water_depth != None:
                
            self.wd['x0'] = float(input('Minimum x-value in domain (m):\n') \
                                    or "0")
            self.wd['x1'] = float(input('Maximum x-value in domain (m):\n') \
                                    or "4000")
            self.wd['dx'] = float(input('Space between depth values (m):\n')\
                                    or "10")

            _ = np.arange(self.wd['x0'], self.wd['x1']+self.wd['dx'],\
                step=self.wd['dx'])

            self.wd['field'] = np.ones_like(_) * constant_water_depth
            self.bathy_defined = True

        else:
            print("No inputs provided. Please specify bathymetry info.")
    
    def define_linear_ssp(self, c_surf=1520., c_slope=-0.1):
        '''
        Establishes linear sound speed profile provided slope and speed at the
        free surface.

        Note that this function does not return outputs but defines class
        attributes critical to the functionality of other attributes.

        Parameters:
        ------------
        c_surf: Sound speed at the surface in m/s. Default value is 1520 m/s.
        c_slope: Rate of change of sound speed with respect to depth. Default
                value is -0.1 (m/s)/m

        Outputs:
        ------------
        None.
        '''

        if self.bathy_defined == False:
            print('Load bathymetry first in order to inform SSP z limits.')
            return

        else:
            #round up so that SSP is defined further than the bathymetry
            ssp_zlim = ceil(np.max(self.wd['field'])/100) * 100.
            dz = float(input('Step size in z-direction (m):\n') or "1")

            #assumes that upper limit of linear SSP is always 0 (sea surface)
            temp_z_arr = np.arange(0, ssp_zlim + dz, step=dz)
            temp_c_arr = c_surf + c_slope * temp_z_arr

            self.ssp['SSP'] = temp_c_arr
            self.ssp['SSP_Z'] = temp_z_arr
        
        self.ssp['SSP'] = np.reshape(self.ssp['SSP'], (1,-1))
        self.ssp_defined = True

    def define_niw_ssp(self, ssp0_c1=1540, ssp0_c2=1500, ssp0_eta0=40,\
                        ssp0_Lc = 10, niw_a=[30,20], niw_b=[40,70], \
                        niw_y0 = [[1e-5, -6e-2, 1000], [1e-5, -6e-2, 1500]],\
                        suppress_niw_packets=False):
        '''
        Establishes  sound speed profile associated with a pair of non-linear
        internal wave packets that disturb the underlying thermocline.

        Note that this function does not return outputs but defines class
        attributes critical to the functionality of other attributes.

        Parameters:
        ------------
        ssp0_c1: Sound speed above the thermocline in meters per second.
        ssp0_c2: Sound speed below the thermocline in meters per second.
        ssp0_eta0: Depth of the background thermocline in meters.
        ssp_Lc: Unknown.
        niw_a: Information about first packet of internal wave front.
        niw_b: Information about second packet of internal wave front.
        niw_y0: Array of information about the internal wave fronts. Last
            value determines the location of the leading edge of the
            respective wave front.
        suppress_niw_packets: Flag to disable internal wave fronts but to keep
            a non-linear thermoclinic sound speed profile.

        Outputs:
        ------------
        None.
        '''
        
        if self.bathy_defined == False:
            print('Load bathymetry first in order to inform SSP z limits.')
            return

        #initialize vars
        niw = {}
        niw['a'] = np.array(niw_a)
        niw['b'] = np.array(niw_b)
        niw['y0'] = np.array(niw_y0)

        ssp0 = {}
        ssp0['c1'] = ssp0_c1
        ssp0['c2'] = ssp0_c2
        ssp0['eta0'] = ssp0_eta0
        ssp0['Lc'] = ssp0_Lc

        self.niw_profile = {}

        #ssp range
        dx = self.wd['dx']
        temp_x_arr = np.arange(self.wd['x0'], self.wd['x1']+dx, step=dx)
        ssp_zlim = np.max(self.wd['field'])
        dz = float(input('Step size in z-direction (m):\n') or "1")

        #assumes that upper limit of linear SSP is always 0 (sea surface)
        temp_z_arr = np.arange(0, ssp_zlim + dz, step=dz)

        #internal function to back out the NIW packet information
        def niw_ssp(ssp0, niw, x, y, z):

            #initialize fxn vars
            detady = 0
            detadx = 0
            eta = ssp0['eta0']

            for _ in range(len(niw['a'])):
                _tanh = np.tanh( (y - np.polyval(np.array(niw['y0'][_]),x)) / \
                    niw['b'][_])
                _sechsq = 1 - _tanh * _tanh
                eta = eta + niw['a'][_]*_sechsq
                detady = detady - 2 * niw['a'][_]/niw['b'][_] * (_tanh*_sechsq)
                detadx = detadx + 2 * niw['a'][_]/niw['b'][_] * \
                    (_tanh*_sechsq) * np.polyval([niw['y0'][_,0]*2, \
                        niw['y0'][_,1]],x)

            if(~np.isnan(z).any()):

                #sound speed profile
                eta = z*np.ones_like(y) - eta*np.ones_like(z)
                _tanh = np.tanh(eta/ssp0['Lc']/2)
                _sechsq = 1 - _tanh * _tanh

                dc = ssp0['c2'] - ssp0['c1']
                c = ssp0['c1'] + dc/2. * (1+_tanh)

                dcdz = dc/4./ssp0['Lc'] * _sechsq
                dcdy = dcdz * -detady * np.ones_like(z)
                dcdx = dcdz * -detadx * np.ones_like(z)
        
            else:

                c = np.nan
                dcdz = np.nan
                dcdy = np.nan
                dcdx = np.nan

            return c, eta, dcdx, dcdy, dcdz, detady

        _ = (len(temp_x_arr), len(temp_z_arr))
        temp_zeros = np.zeros_like(temp_x_arr)

        if suppress_niw_packets == False:
            self.niw_profile['field'] = np.empty(_)
            self.niw_profile['eta'] = np.empty(_)
            self.niw_waveform = True
            self.ssp['SSP'] = np.empty(_)

            for idz in np.arange(0, len(temp_z_arr)):
            
                self.niw_profile['field'][:,idz] = niw_ssp(ssp0, niw, \
                    temp_zeros, temp_x_arr, temp_z_arr[idz])[0]
                self.niw_profile['eta'][:,idz] = niw_ssp(ssp0, niw, \
                    temp_zeros, temp_x_arr, np.nan)[1]
                self.ssp['SSP'][:,idz] = niw_ssp(ssp0, niw, \
                    temp_zeros, temp_x_arr, temp_z_arr[idz])[0]

            #self.ssp['SSP']=niw_ssp(ssp0, niw, self.source['x'], \
            #    0, temp_z_arr)[0]
        else:
            self.ssp['SSP'] = niw_ssp(ssp0, niw, self.source['x'], 0, \
                temp_z_arr)[0]
            self.ssp['SSP'] = np.reshape(self.ssp['SSP'], (1,-1))
        
        self.ssp['SSP_Z'] = temp_z_arr
        #self.ssp['SSP'] = np.reshape(self.ssp['SSP'], (1,-1))
        print(self.ssp['SSP'].shape)
        self.ssp_defined = True

    def define_slowness(self):
        '''
        Create the slowness field as the inverse of the sound speed profile.
        Assumes that a sound speed profile has already been defined.
        '''

        if self.ssp_defined:
            self.slowness = deepcopy(self.ssp)
            self.slowness['SSP'] = 1. / self.ssp['SSP']
            #self.slowness['SSP'] = np.reshape(self.slowness['SSP'], (1,-1))
        else:
            print('Must define sound speed profile first.')

    def define_receiver(self, receiver_loc=[None, None]):
        '''
        Establish attributes for receiver location
        
        Parameters:
        ------------
        receiver_loc: x, z coordinates of the receiver location. If no
            coordinates are specified, this will assume a receiver located
            at the maximum x location in the domain at the same depth as the
            source.
        '''

        if receiver_loc == [None, None]:
            receiver_loc = [self.wd['x1'], self.source['z']]

        self.rcvr={}
        self.rcvr['x'] = receiver_loc[0]
        self.rcvr['z'] = receiver_loc[1]

    def define_source(self, source_location=[0,20]):
        '''
        Establish attributes for source location
        
        Parameters:
        ------------
        source_location: x, z coordinates of the source location. If no
            coordinates are specified, this will assume a source located at 
            x = 0 and a depth of 20 meters below the sea surface.
        '''

        self.source={}
        self.source['x'] = source_location[0]
        self.source['z'] = source_location[1]

    def find_eigenrays(self, receiver_loc=[None, None],\
        theta_range=[-10,-5,0,5,10]):
        '''
        Establishes eigenrays for a provided source and receiver location
        in the defined environment. Eigenrays are those rays that pass directly
        through the receiver location. Eigenrays are determined using simple
        zero-crossing refinement algorithms.

        This function does not have any explicit outputs, but does define the
        attributes of self.eigenangles and self.eigentaus, which specify the
        zero-crossing angles and the maximum ray travel time, respectively.

        Parameters:
        --------------
        receiver_loc: x, z coordinates of the receiver location. If no
            coordinates are specified, this will assume a receiver located
            at the maximum x location in the domain at the same depth as the
            source.
        theta_range: Array of initial ray takeoff angles (in degrees) to 
            iterate through in order to determine eigenrays and eigenangles. 
            The default range is from -10 to 10 degrees.

        Outputs:
        -------------
        None.
        '''

        self.define_receiver(receiver_loc)

        #parallelize the calculation of the ray end points
        t = time.time()
        temp=np.array(Parallel(n_jobs=-1)(delayed(self._get_xfan)(i) \
            for i in theta_range), dtype=object)
        z_fan = np.asarray([az[-1] for az in temp[:,1]])

        zero_crossing = z_fan - self.rcvr['z']
        zero_crossing = zero_crossing[:-1] * zero_crossing[1:]
        eigenangle = np.argwhere(zero_crossing<0)
        eigenangle = np.array([eigenangle, eigenangle+1]).squeeze()

        #refine eigenangle - using self since need to reference in outside fxn
        self.zero_crossing0 = (z_fan[eigenangle[0]].T-self.rcvr['z']).squeeze()
        self.zero_crossing1 = (z_fan[eigenangle[1]].T-self.rcvr['z']).squeeze()
        self.angle0 = theta_range[eigenangle[0]].T.squeeze()
        self.angle1 = theta_range[eigenangle[1]].T.squeeze()

        temp2 = np.array(Parallel(n_jobs=-1)(delayed(self._refine_angle)(i)\
            for i,_ in enumerate(eigenangle[0])), dtype=object)
        
        self.eigenangles = np.array([min(ai) for ai in temp2[:,0]])
        self.eigentaus = np.array([max(ai) for ai in temp2[:,-1]])

        print(f'Eigenangles found in {time.time()-t:.2f} sec')

    def plot_rays(self, theta_list, savename=None, matlabfile=None,\
        abstol=1e-9, reltol=1e-6, nmaxbtmbnce=10, ds=10, ode_method='RK45',
        use_max_ds=False):
        '''
        Generates a plot of rays in specified environment. Utilizes environment
        data and source location previously defined by other class attributes.

        This function does not have any explicit outputs, but does write the
        output plot to a file for future viewing.

        Parameters:
        --------------
        theta_list: Array of initial ray take-off angles in degrees for which
            to create ray traces.
        savename: Name of output image file. Must include full path to save
            location.
        matlabfile: Array-like of Matlab .mat files of ray_x and ray_z for use
            in overlaying ray traces from Matlab codes with Python results.
        abstol: Absolute tolerance for numerical error in the ODE solver.
            Default value is 1.e-9.
        reltol: Relative tolerance for numerical error in the ODE solver.
            Default value is 1.e-6.
        ds: Maximum step size allowed for the Runge-Kutta marching algorithm.
            Measured in meters along the arclength of the resultant ray.
            Default value is 10.
        nmaxbtmbnce: Maximum number of bottom bounces before ray is terminated.
            Default value is 10.
        ode_method: Specifies solve_ivp ODE marching method. Valid inputs are
            RK23, RK45, DOP853, and LSODA (in order of increasing order and
            accuracy). Default value is RK45.
        use_max_ds: Flag to disable the use of maximum ds step in solve_ivp.
            In order to most closely, match Matlab results and minimize
            numerical errors, this should be set to True with a ds as stated
            above.

        Outputs:
        -------------
        None.
        '''
        theta_list = np.asarray(theta_list)

        #initialize figure object
        screen_x, screen_y = pyautogui.size()
        px = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=[screen_x/px, screen_y/px])
        window = plt.get_current_fig_manager().window
        newX = screen_x//10
        newY = screen_y//10
        dx = screen_x//5.*4
        dy = screen_y//5.*4
        window.setGeometry(int(newX), int(newY), int(dx), int(dy))
        
        #plot sound speed profile in separate figure
        zlim = np.max(self.ssp['SSP_Z'])+10
        ax = fig.add_axes([1.2/20, 1/2-0.05,2.5/20,1/2-0.05])
        plt.rcParams.update({'font.size': 18})
        ax.set_ylim([0, zlim])
        xmin, xmax = np.min(self.ssp['SSP'])-5, np.max(self.ssp['SSP'])+5
        ax.set_xlim([xmin, xmax])
        ax.xaxis.set_ticks(np.linspace(np.min(self.ssp['SSP']), \
            np.max(self.ssp['SSP']), num=2))
        ax.yaxis.set_ticks(np.linspace(0, np.max(self.ssp['SSP_Z']), num=11))
        ax.invert_yaxis()
        ax.set_ylabel('Z (m)', fontsize=16)
        ax.set_xlabel('Sound Speed (m/s)', fontsize=20)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.yaxis.set_label_position('left') 

        #plot ssp
        xs, zs = np.median(self.ssp['SSP']), self.source['z']
        ax.plot(xs, zs, 'k*', markersize=16)
        ax.plot(xs, zs, 'ko', markersize=8)
        ax.text(xs, zs, '  Point Source', fontsize=16,\
            horizontalalignment='left', verticalalignment='center')
        if len(self.wd['field']) == 1:
            wd = ax.hlines(np.max(self.ssp['SSP_Z']), xmin, xmax, 'k',\
                linewidth=2)
        else:
            _ = np.linspace(xmin, xmax, num=len(self.wd['field'])-1)
            ax.plot(_, np.max(self.ssp['SSP_Z'])*np.ones_like(_), 'k')
        test_patch = Rectangle((xmin, np.max(self.ssp['SSP_Z'])),\
            width=xmin-xmax, height=10, linestyle=None)
        ax.add_patch(test_patch)

        try:
            ax.plot(self.ssp['SSP'].squeeze(), self.ssp['SSP_Z'], 'k')
        except:
            try:
                ax.plot(self.ssp['SSP'], self.ssp['SSP_Z'], 'k')
            except:
                ax.plot(self.ssp['SSP'][0], self.ssp['SSP_Z'], 'k')

        #ray tracing setup
        ax2 = plt.axes([1/5+.1, 0+.2, 3/4*.8, .7])
        xmin, xmax = self.wd['x0'], self.wd['x1']
            
        ax2.plot([xmin, xmax], [0,0], 'k', linewidth=3)
        if len(self.wd['field']) == 1:
            wd = ax2.hlines(np.max(self.ssp['SSP_Z']), xmin, xmax, \
                'k', linewidth=2)
        else:
            _ = np.arange(self.wd['x0'], self.wd['x1']+self.wd['dx'], \
                step=self.wd['dx'])
            ax2.plot(_, self.wd['field'], 'k')
        ax2.set_xlabel('X (m)', fontsize=20)
        ax2.set_ylabel('Z (m)', fontsize=20)
        ax2.plot(self.source['x'], self.source['z'], 'k*', markersize=16)
        ax2.plot(self.source['x'], self.source['z'], 'ko', markersize=8)
        ax2.set_xlim([xmin, xmax])
        #ax2.set_xlim([6000, 7000]) to check blue
        zlim = np.max(self.ssp['SSP_Z'])+10
        ax2.set_ylim([-zlim*0.05, zlim])
        ax2.invert_yaxis()

        colors = cm.gist_rainbow(np.linspace(0,1,num=len(theta_list)+1))

        if self.niw_waveform:
            surfx = np.linspace(self.wd['x0'], self.wd['x1'], \
                num=self.niw_profile['eta'][:,0].size)
            ax2.plot(surfx, self.niw_profile['eta'][:,0], 'k', linewidth=3)

        for j, atheta in enumerate(theta_list):
        
            x,z,*_ = self.create_ray_trace(atheta, abstol=abstol, \
                reltol=reltol, nmaxbtmbnce=nmaxbtmbnce, \
                    ds=ds, ode_method=ode_method, use_max_ds=use_max_ds)

            #plot the ray
            ax2.plot(x, z, color=colors[j], linewidth=2, label=f'{atheta} deg')
            ax2.plot(x[-1],z[-1], 'ko', markersize=8)
        
            if matlabfile != None:
                if type(matlabfile) != list:
                    matlabfile = [matlabfile]
                _ = loadmat(matlabfile[j])
                ax2.plot(_['ray_x'], _['ray_z'], color='k', linestyle='--', \
                    dashes=(1, 1), linewidth=2)

        #having issues with IDE using plt.show() so going to save 
        #and view image instead.
        if True:
            px = plt.rcParams['figure.dpi']
            if savename == None:
                plt.savefig('test.png', dpi=px)
            else:
                plt.savefig(savename, dpi=px)
            plt.close()

    def read_ssp_from_file(self, fname):
        '''Read the sound speed profile from a Matlab .mat file
        
        ***FUTURE WORK MAY BE TO READ THIS AS A 2D FIELD RATHER THAN A 
        STATIONARY PROFILE AT INLET***

        Parameters:
        ------------
        fname: Filename of Matlab .mat file from which to read the ambient
            sound speed profile. Assumes that .mat file has a singular depth
            profile that is taken to be valid across entire domain.

                SSP: Value of sound speed in meters per second.
                SSP_Z: Values of depth in meters for each sound speed value.
        '''

        self.ssp = loadmat(fname)
        self.ssp['SSP'] = self.ssp['SSP'].flatten()
        self.ssp['SSP'] = np.reshape(self.ssp['SSP'], (1,-1))
        self.ssp['SSP_Z'] = self.ssp['SSP_Z'].flatten()
        self.ssp_defined = True
