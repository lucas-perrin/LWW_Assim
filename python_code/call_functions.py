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

def Matrix_A_dz(Nx, dx):
    k = 2 * np.pi * fftfreq(Nx, dx)
    kernel = np.real(np.fft.ifft(np.abs(k))) * Nx
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