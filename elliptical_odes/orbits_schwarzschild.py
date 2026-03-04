# %%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
def r_s(M = 1.0):
    """
    Compute Schwarzschild Radius
    
    float M: Object mass in Msun units
    returns float: Schwarzschild Radius
    """
    return 2.*Mlen*M

def half_r_s(M= 1.0):
    """
    Compute Half-Schwarzschild Radius
    
    float M: Object mass in Msun units
    returns float: Half-Schwarzschild Radius
    """    
    return Mlen*M

def gtt_schw(r):
    """
    Compute gtt for Schwarzschild
    
    float r: radius
    returns float: gtt 
    """
    return 1.0 - r_s(M)/r

def rhs(tau, y, E, L, M=1.0):
    """
    Function for runge-kutta step.

    float tau: proper time
    array (float) y: [coordinate time, radius, angle phi, radial velocity]
    float E, L: energy and angular momentum of the system
    float M: object mass in Msun units

    returns:
    array (float) : dt/dtau, dr/dtau, dphi/dtau, dvr/dtau
    """
    t, r, phi, vr = y

    f = gtt_schw(r)

    dt_dtau = E / f

    dvr_dtau = L*L/(r**3) - half_r_s(M)/r**2 - 3*half_r_s(M)*L*L/r**4
    dr_dtau = vr
    
    dphi_dtau = L / r**2

    return np.array([
        dt_dtau,
        dr_dtau,
        dphi_dtau,
        dvr_dtau
    ])


def runge_kutta(tau, y, h, *args):
    """
    Compute runge_kutta step
    
    float tau: proper time
    array (float) y: [coordinate time, radius, angle phi, radial velocity]
    float h: timestep
    float *args: [E, L, M]

    returns array (float): evolved step yn+1 [coordinate time, radius, angle phi, radial velocity]
    """
    k1 = rhs(tau, y, *args)
    k2 = rhs(tau + h/2, y + h*k1/2, *args)
    k3 = rhs(tau + h/2, y + h*k2/2, *args)
    k4 = rhs(tau + h,   y + h*k3,   *args)

    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6


def find_perihelia(t, r, phi):
    """
    Find perihelium at every orbit by looking at radius inversion

    float t: coordinate time
    float r: radius
    float phi: angle

    returns: indeces of perihelia
    """
    return np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1

def compute_mean_perihelia(t,r,phi, idx):
    """
    Compute the mean precession from a set of perihelia.
    
    float t: time
    float r: radius
    float phi: angles
    array (int) idx: indeces of perihelia
    """
    phi = phi % (2*np.pi)   #write phi in interval between [0, 2pi]
    dphi = np.diff(phi[idx])
    return np.mean(dphi)

# %%
#constants
year_conversion = 3.156e7   # s in one year
c = 299792458.0             # m/s
G = 6.67430e-11             # m^3/(kg s^2)
Msun = 1.98847e30           # kg
M = 1.                      # Msun
Mlen = G*Msun/(c*c)         # Schwarzschild mass parameter in meters (GM/c^2)

# %%
#Mercury values
a = 5.7909227e10            #semi major axis
e = 0.20563                 #eccentricity
M_mercury = 3.3e23          #Mercury Mass (Msun)

rp = a*(1.0-e)              #perihelion distance
ra = a*(1.0+e)              #aphelion distance

period = 0.240846           #orbit period

#Derive E and L
gtt_rp = gtt_schw(rp)
gtt_ra = gtt_schw(ra)

L_squared = (gtt_ra-gtt_rp)/(gtt_rp/rp**2-gtt_ra/ra**2)
E_squared = gtt_ra*(1+L_squared/ra**2)
E_squared = gtt_rp*(1+L_squared/rp**2)

L = np.sqrt(L_squared)
E = np.sqrt(E_squared)
print('-------------  Check L, E -------------')
print('L squared =', L_squared)
E_squared = gtt_ra*(1+L_squared/ra**2)
print('E squared from ra =', E_squared)
E_squared = gtt_rp*(1+L_squared/rp**2)
print('E squared from rp =', E_squared)

print('Energy:', E)
print('Angular Momentum:', L)

# %%
#initial conditions
tau_0 = 0
r_0 = rp
t_0 = 0
phi_0 = 0
vr_0 = 0

N_orbits = 50.#10/period
T = N_orbits*2*np.pi*np.sqrt(a**3/(half_r_s(M)))   # ~ 7.6e6 s for Mercury

# Choose steps
N = int(1e9)                           
h = T/N

# %%
y = np.array([t_0, r_0, phi_0, vr_0])

traj = np.zeros((N, 4))
tau = tau_0
tau_list = []
tau_list.append(tau)
for i in range(N):
    traj[i] = y
    y = runge_kutta(tau, y, h, E, L)
    tau += h
    #tau_list.append(tau)

#tau = np.array(tau_list)   #for memory purposes

# %%
t = traj[:, 0]/c
r   = traj[:,1]
phi = traj[:,2]
vr = traj[:, 3]

del traj #for memory purposes

# %%
idx_perihelia = find_perihelia(t,r,phi)
dphi = compute_mean_perihelia(t,r,phi,idx_perihelia)
print('Precession per orbit (radians):', dphi)

N_orbits_century = 100/period
rad_to_arcsec = 180/np.pi * 3600
print('Precession after a century:', dphi*N_orbits_century*rad_to_arcsec)

r = r/1e10
plt.figure()
ax = plt.subplot(111, projection='polar')
phi_unwrapped = np.unwrap(phi)
ax.plot(phi_unwrapped, r)
ax.set_title("Mercury Orbit with Precession")
plt.savefig('polar_plot.png', dpi = 300)
plt.show()


np.savez('orbit_data.npz', t=t, r=r, phi=phi, vr=vr, idx_perihelia=idx_perihelia)