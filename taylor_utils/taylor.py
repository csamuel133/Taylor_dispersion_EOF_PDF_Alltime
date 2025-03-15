import numpy as np
import seaborn as sns
import scipy
from scipy.special import i0, i1
import pickle

def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None

def weighted_mean_r_moment(x):
    return np.sum(x, axis = 0, keepdims = True)/np.sum(np.ones_like(x), axis = 0, keepdims = True)
def weighted_variance_r_moment(x, weighted_x):
    return np.sum((x-weighted_x)**2, axis = 0)/np.sum(np.ones_like(x), axis = 0)

def simulation_moment_EOFPoiseuille_initialVar(Pe0, phi, beta, Nt0 = 500, seed = 0, sigx2_0 = 300, 
                                                        upper_bound = None, A = 40):
    # define parameter
    # phi is radius/Debye length
    U0 = 1 # <u_x>
    Up = U0*beta # <u_p>
    Ue = U0 - Up # <u_e>
    eta = 2/phi*i1(phi)/i0(phi)
    u_eo = Ue/(1-eta)    
    a0 = 1
    B = A/Pe0
    dt = 1/B
    D = B/A
    Npts = 5000
    Nt = Nt0*A+1
    sig_s = np.sqrt(2*D*dt)
    np.random.seed(seed)

    # parameters related to analysis
    gamma_p = 1/48
    gamma_pe = eta/(1-eta)*(1/12 - 2/phi**2 + 16/phi**4*(1-eta)/eta)
    gamma_e = (eta/(1-eta))**2*(3/8 + 2/phi**2 - 1/(eta*phi**2) - 1/(eta**2*phi**2))
    
    # initialization
    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    x[:, 0] = np.random.randn(Npts)*np.sqrt(sigx2_0)
    theta[:, 0] = (np.random.rand(Npts))*2*np.pi - np.pi
    r[:,0] = np.sqrt((np.random.rand(Npts)))
    
    if upper_bound:
        x_range_est = upper_bound
    else:
        x_range_est = Nt*dt*U0*8
        
    ux = lambda x, r: 2*Up*(1 - r**2) + u_eo*(1 - i0(phi*r)/i0(phi))
    ur = lambda x, r: 0
    
    rand = np.random.randn(Nt-1, 3*Npts)

    # simulation
    for i in range(1, Nt):
        x[:, i] = x[:, i-1] + ux(x[:, i-1], r[:, i-1])*dt + sig_s*rand[i-1, 0:Npts]
        r_temp = r[:, i-1] + ur(x[:, i-1], r[:, i-1])*dt
        x2 = r_temp*np.cos(theta[:, i-1]) + sig_s*rand[i-1, Npts:2*Npts]
        x3 = r_temp*np.sin(theta[:, i-1]) + sig_s*rand[i-1, 2*Npts:3*Npts]
        theta[:, i] = np.arctan2(x3, x2)
        r_new = np.sqrt(x2**2 + x3**2)
        loc_pos = (r_new > 1)
        r_new[loc_pos] = 2*1 - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        loc_pos = (r_new > 1)
        r_new[loc_pos] = 2*1 - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        r[:, i] = r_new

    # analysis
    T = np.arange(Nt)*dt
    weighted_x = weighted_mean_r_moment(x)
    weighted_var = weighted_variance_r_moment(x, weighted_x)

    result = {'x': x,
              'r': r,
              'theta': theta,
              'T': T,
              'weighted_x': weighted_x, 
              'weighted_var': weighted_var,
              'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt, 'Npts': Npts, 'Nt': Nt,
              'eta': eta, 'beta': beta, 'phi': phi, 
              'gamma_p': gamma_p, 'gamma_pe': gamma_pe, 'gamma_e': gamma_e}
    return result

def simulation_moment_EOFPoiseuille_initialVar_zeroPe(Ue, Up, D, phi, Nt0 = 500, seed = 0, sigx2_0 = 300, 
                                                        upper_bound = None, sampling_ratio = 100):
    a0 = 1
    U0 = Ue + Up
    beta = np.divide(Up,U0)
    Pe0 = U0*a0/D
    eta = 2/phi*i1(phi)/i0(phi)
    u_eo = Ue/(1-eta)    
    A = 5000
    dt = a0**2/D/A
    
    Npts = 5000
    Nt = int(Nt0*A/sampling_ratio+1)
    sig_s = np.sqrt(2*D*dt)
    np.random.seed(seed)

    # parameters related to analysis
    gamma_p = 1/48
    gamma_pe = eta/(1-eta)*(1/12 - 2/phi**2 + 16/phi**4*(1-eta)/eta)
    gamma_e = (eta/(1-eta))**2*(3/8 + 2/phi**2 - 1/(eta*phi**2) - 1/(eta**2*phi**2))
    
    # initialization
    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    
    x[:, 0] = np.random.randn(Npts)*np.sqrt(sigx2_0)
    theta[:, 0] = (np.random.rand(Npts))*2*np.pi - np.pi
    r[:,0] = np.sqrt((np.random.rand(Npts)))
    
    if upper_bound:
        x_range_est = upper_bound
    else:
        x_range_est = Nt*dt*U0*8
        
    ux = lambda x, r: 2*Up*(1 - r**2) + u_eo*(1 - i0(phi*r)/i0(phi))
    ur = lambda x, r: 0

    # simulation
    for i in range(1, Nt):
        r_samp = np.zeros((Npts, sampling_ratio+1))
        theta_samp = np.zeros((Npts, sampling_ratio+1))
        x_samp = np.zeros((Npts, sampling_ratio+1))
        x_samp[:, 0] = x[:, i-1]
        theta_samp[:, 0] = theta[:, i-1]
        r_samp[:,0] = r[:, i-1]
        rand = np.random.randn(sampling_ratio, 3*Npts)
        for j in range(1, sampling_ratio+1):
            x_samp[:, j] = x_samp[:, j-1] + ux(x_samp[:, j-1], r_samp[:, j-1])*dt + sig_s*rand[j-1, 0:Npts]
            r_temp = r_samp[:, j-1] + ur(x_samp[:, j-1], r_samp[:, j-1])*dt
            x2 = r_temp*np.cos(theta_samp[:, j-1]) + sig_s*rand[j-1, Npts:2*Npts]
            x3 = r_temp*np.sin(theta_samp[:, j-1]) + sig_s*rand[j-1, 2*Npts:3*Npts]
            theta_samp[:, j] = np.arctan2(x3, x2)
            r_new = np.sqrt(x2**2 + x3**2)
            loc_pos = (r_new > 1)
            r_new[loc_pos] = 2*1 - r_new[loc_pos]
            loc_neg = (r_new < 0)
            r_new[loc_neg] = -r_new[loc_neg]
            theta[loc_neg, j] = theta[loc_neg, j] + np.pi
            loc_pos = (r_new > 1)
            r_new[loc_pos] = 2*1 - r_new[loc_pos]
            loc_neg = (r_new < 0)
            r_new[loc_neg] = -r_new[loc_neg]
            theta[loc_neg, j] = theta[loc_neg, j] + np.pi
            r_samp[:, j] = r_new
        x[:, i] = np.copy(x_samp[:, j])
        theta[:, i] = np.copy(theta_samp[:, j])
        r[:, i] = np.copy(r_samp[:, j])

    # analysis
    T = np.arange(Nt)*dt*sampling_ratio
    weighted_x = weighted_mean_r_moment(x)
    weighted_var = weighted_variance_r_moment(x, weighted_x)

    result = {'x': x,
              'r': r,
              'theta': theta,
              'T': T,
              'weighted_x': weighted_x, 
              'weighted_var': weighted_var,
              'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt, 'Npts': Npts, 'Nt': Nt,
              'eta': eta, 'Up': Up,'Ue' : Ue, 'phi': phi, 
              'gamma_p': gamma_p, 'gamma_pe': gamma_pe, 'gamma_e': gamma_e}
    return result