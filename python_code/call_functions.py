import numpy as np

##################################################

def del_hig_freqs(frequencies, thrs):
    DF = np.diag(np.abs(frequencies) <= thrs) * (np.abs(frequencies) > 0)
    k = 0
    while k < DF.shape[0]:
        if np.sum(DF[k, :]) == 0:
            DF = np.delete(DF, k, axis=0)
        else:
            k = k + 1
    return DF

##################################################

def del_low_freqs(frequencies, thrs):
    DF = np.diag(np.abs(frequencies) > thrs) * (np.abs(frequencies) > 0)
    k = 0
    while k < DF.shape[0]:
        if np.sum(DF[k, :]) == 0:
            DF = np.delete(DF, k, axis=0)
        else:
            k = k + 1
    return DF

##################################################

def fftfreq(n, d):
    if n % 2:
        k = np.concatenate((np.arange(0, (n - 1) / 2 + 1), np.arange(-(n - 1) / 2, 0))) / (d * n)
    else:
        k = np.concatenate((np.arange(0, n / 2), np.arange(-n / 2, 0))) / (d * n)
    return k

##################################################

def Matrix_A_dz(Nx, dx, d):
    k = 2 * np.pi * fftfreq(Nx, dx)
    kernel = np.real(np.fft.ifft(np.abs(k)*np.tanh(d*np.abs(k)))) * Nx
    A_dz = np.zeros((Nx, Nx))
    for j in range(Nx):
        A_dz[j, :] = (1 / Nx) * np.roll(kernel, j)
    return A_dz

##################################################

def GetC(obs_map):
    C_flat = obs_map.flatten()
    nb_obs = np.sum(C_flat != 0)
    C = np.zeros((nb_obs, len(C_flat)))
    j = 0
    for p in range(len(C_flat)):
        if C_flat[p] != 0:
            C[j, p] = C_flat[p]
            j = j + 1
    return nb_obs, C

##################################################

def getDFT(N):
    w_N = np.exp(-2j * np.pi / N)
    W_N = w_N ** (np.outer(np.arange(N), np.arange(N)))
    return W_N

##################################################

def getiDFT(N):
    w_N = np.exp(2j * np.pi / N)
    W_N = w_N ** (np.outer(np.arange(N), np.arange(N)))
    return W_N

##################################################

def getDFT_no_mean(N):
    w_N = np.exp(-2j * np.pi / N)
    W_N = w_N ** (np.outer(np.arange(N)[1:], np.arange(N)))
    return W_N

##################################################

def getiDFT_no_mean(N):
    w_N = np.exp(2j * np.pi / N)
    W_N = w_N ** (np.outer(np.arange(N), np.arange(N)[1:]))
    return W_N

##################################################

# Euler ODE does not work (yet) when called in the notebook, but work when implemented in the notebook. To investigate.

def Euler_Explicit(mat, X0, tspan):
    X = np.zeros((X0.shape[0],tspan.shape[0])) + np.zeros((X0.shape[0],tspan.shape[0]))*1j
    dt = tspan[1] - tspan[0]
    X[:,0] = X0
    for i in range(1,len(tspan)):
        X[:,i] = X[:,i-1] + dt * mat @ X[:,i-1]
    return X

##################################################

def Euler_Implicit(mat, X0, tspan):
    X = np.zeros((X0.shape[0],tspan.shape[0])) + np.zeros((X0.shape[0],tspan.shape[0]))*1j
    dt = tspan[1] - tspan[0]
    X[:,0] = X0
    for i in range(1,len(tspan)):
        X[:,i] = np.linalg.solve((1 - dt * mat),X[:,i-1])
    return X

##################################################

def create_Fourier_X0(nb_sin, nb_cos, Nx, dx):
    frequences = fftfreq(Nx, dx) * 2 * np.pi
    pos_frequences = frequences[1:int((len(frequences)-1)/2)+1]
    sin_pos = np.random.rand(pos_frequences.shape[0]) * (pos_frequences <= nb_sin)
    cos_pos = np.random.rand(pos_frequences.shape[0]) * (pos_frequences <= nb_cos)
    X0 = np.concatenate((np.zeros(1),-sin_pos,sin_pos[::-1]))*1j + np.concatenate((np.zeros(1),cos_pos,cos_pos[::-1]))
    return X0

##################################################

def create_Fourier_X0_v2(nb_freqs_ini, Nx, dx):
    frequences = fftfreq(Nx, dx) * 2 * np.pi
    X0 = (np.abs(frequences) < nb_freqs_ini+1) * (np.abs(frequences) > 0) *(np.random.rand(frequences.shape[0]) + 1j * np.random.rand(frequences.shape[0]))
    return X0

##################################################

def largest_nonzero_eigenvalue(matrix):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eig(matrix)

    # Filter out nonzero eigenvalues
    nonzero_eigenvalues = eigenvalues[eigenvalues != 0]

    if len(nonzero_eigenvalues) == 0:
        # If there are no nonzero eigenvalues, return None
        return None

    # Find the largest nonzero eigenvalue
    largest_nonzero_eigenvalue = np.max(np.real(nonzero_eigenvalues))

    return largest_nonzero_eigenvalue

##################################################

def H_half_norm(vect,frequences):
    return np.abs(frequences) @ (vect**2)

##################################################

def fH_half_times_L2_norm(vect,frequences):
    nb_frequences = frequences.shape[0]
    phi = vect[0:nb_frequences,:]
    eta = vect[nb_frequences:,:]
    return (np.abs(frequences) @ (phi**2))**2 + np.linalg.norm(eta,2,axis=0)**2

##################################################

def H_half_times_L2_norm(vect,Nx,dx):
    frequences = fftfreq(Nx, dx) * 2 * np.pi
    DFT = 1 / Nx * getDFT(Nx)
    iDFT = getiDFT(Nx)
    phi = DFT @ vect[0:Nx,:]
    eta = DFT @ vect[Nx:,:]
    return ((np.abs(frequences) @ np.abs(phi**2)) + np.linalg.norm(eta,2,axis=0))**(1/2)