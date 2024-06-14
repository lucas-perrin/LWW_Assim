import numpy as np
import matplotlib.pyplot as plt

from call_functions import *

Fontsize = 18
Fontsize_label = 20
Fontsize_axes = 18
Linesize = 2
Marksize = 9

# ------------------------------
# Parameters to tune
# ------------------------------

x0 = 0
L = 2 * np.pi
Nx = 2**8
Nf = Nx // 2
dt = 1e-3
a = 1
g = 1
gain = 1
T_start = 0
T = 500
N_ini_fq = 2**2

# ------------------------------
# Space discretization
# ------------------------------

dx = L / Nx
Lx = L - dx
xspan = np.arange(x0, Lx + dx, dx)
frequences = np.fft.fftfreq(Nx, dx) * 2 * np.pi

# ------------------------------
# Time discretization
# ------------------------------

T_end = T_start + T
tspan = np.arange(T_start, T_end + dt, dt)
Nt = len(tspan)

# ------------------------------
# Matrix computations
# ------------------------------

# 1) DFT and iDFT Matrices (as well as no mean matrices)
DFT = 1 / Nx * getDFT(Nx)
iDFT = getiDFT(Nx)

DFT_n_m = 1 / Nx * getDFT_no_mean(Nx)
iDFT_n_m = getiDFT_no_mean(Nx)

# 2) True matrix A not in Fourier
O = np.zeros((Nx, Nx))
I = np.eye(Nx)
G = Matrix_A_dz(Nx, dx)
_, C = GetC(np.abs(xspan - np.pi) <= a)
obs_space = np.abs(xspan - np.pi) <= a
C = -np.dot(C.T, C)
A = np.block([[O, -I], [G, C]])

# True matrix A in Fourier
fO = np.zeros((2 * Nf + 1, 2 * Nf + 1))
fI = -np.eye(2 * Nf + 1)
fF = np.diag(np.abs(np.concatenate((np.arange(0, Nf + 1), np.arange(-Nf, 0)))) * 2 * np.pi)
frequences = fftfreq(Nx, dx) * 2 * np.pi  # Assuming you have previously defined fftfreq function
fC = np.real(-((a * np.exp((-1j * np.pi) * (frequences.reshape(-1, 1) - frequences))) / np.pi) * np.sinc(a * (frequences.reshape(-1, 1) - frequences) / np.pi))
fA = np.block([[fO, fI], [fF, fC]])

# 3) computing the pseudo matrix A (size (Nx-1)^2) :

# getting A in Fourier : DFT * A * iDFT :
A_f = np.kron(np.eye(2), DFT) @ A @ np.kron(np.eye(2), iDFT)

# removing the mean columns and rows :
A_f_11 = A_f[:Nx, :Nx]
A_f_12 = A_f[:Nx, Nx:2 * Nx]
A_f_21 = A_f[Nx:2 * Nx, :Nx]
A_f_22 = A_f[Nx:2 * Nx, Nx:2 * Nx]
A_fr = np.block([[A_f_11[1:Nx, 1:Nx], A_f_12[1:Nx, 1:Nx]],
                    [A_f_21[1:Nx, 1:Nx], A_f_22[1:Nx, 1:Nx]]])

# getting back in space with a pseudo DFT :
A_fr_s = np.kron(np.eye(2), iDFT[1:Nx, 1:Nx]) @ A_fr @ np.kron(np.eye(2), DFT[1:Nx, :Nx])
A_fr_s_2 = np.kron(np.eye(2), iDFT[1:Nx, 1:Nx]) @ A_fr @ np.kron(np.eye(2), DFT[1:Nx, 1:Nx])

# obtaining the convergence factor for this matrix :
eig_A_fr_s = np.linalg.eigvals(A_fr_s)
eig_A_fr_s = eig_A_fr_s[np.abs(eig_A_fr_s) > 1e-10]
conv_fact_fr_s = min(np.abs(eig_A_fr_s))

# eigenvalues of A :
V, D = np.linalg.eig(A)
eig_A = np.diag(D)
cfactor = min(np.abs(np.real(eig_A[np.abs(np.imag(eig_A)) > 1e-10])))
place_cfactor = np.argmin(np.abs(np.real(eig_A)))
eig_cfactor = eig_A[place_cfactor]

# rojection on N_ini_fq :
DFh = del_hig_freqs(frequences, N_ini_fq)
A_ini_f = np.kron(np.eye(2), DFh @ DFT) @ A @ np.kron(np.eye(2), iDFT @ DFh.T)
A_ini_s = np.kron(np.eye(2), DFh @ iDFT) @ A @ np.kron(np.eye(2), DFT @ DFh)

# eigenvalues of A_ini_s :
V_ini_s, D_ini_s = np.linalg.eig(A_ini_s)
eig_A_ini_s = np.diag(D_ini_s)
cfactor = min(np.abs(np.real(eig_A_ini_s[np.abs(np.imag(eig_A_ini_s)) > 1e-10])))
place_cfactor_ini_s = np.argmin(np.abs(np.real(eig_A_ini_s)))
eig_cfactor_ini_s = eig_A_ini_s[place_cfactor_ini_s]

# Plot the eigenvalues
plt.figure(1, figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(np.real(eig_A_ini_f), np.imag(eig_A_ini_f), 'x',
            np.real(eig_A_fr), np.imag(eig_A_fr), 'o', markersize=Marksize, linewidth=Linesize)
plt.legend(['A f', 'A fr'])
plt.grid()
plt.title('Fourier')

plt.subplot(1, 2, 2)
plt.plot(np.real(eig_A), np.imag(eig_A), 'x',
            np.real(eig_A_fr_s), np.imag(eig_A_fr_s), 'o',
            np.real(eig_A_fr_s_2), np.imag(eig_A_fr_s_2), 's',
            np.real(eig_A_ini_s), np.imag(eig_A_ini_s), '^',
            np.real(eig_cfactor), np.imag(eig_cfactor), 'd', markersize=Marksize, linewidth=Linesize)
plt.legend(['A', 'A fr s', 'A fr s 2'])
plt.grid()
plt.title('Space')

plt.show()