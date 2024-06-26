% ------------------------------
% Parameters to tune
% ------------------------------

x0 = 0;
L = 2 * pi;
Nx = 2^6 + 1;
Nf = floor(Nx / 2);
d = 1000 * pi;
dt = 1e-3;
a = 1/2 * pi;
g = 9.80665;
gain = 1;
T_start = 0;
T = 200;
N_ini_fq = 2^1;

% ------------------------------
% Space discretization
% ------------------------------

dx = L / Nx;
Lx = L - dx;
xspan = x0:dx:Lx;
frequences = fftshift((0:Nx-1) / Nx / dx) * 2 * pi;
frequences2 = fftshift((0:Nx-1) / Nx / dx) * 2 * pi; % same as above

% ------------------------------
% Time discretization
% ------------------------------

T_end = T_start + T;
tspan = T_start:dt:T_end;
Nt = length(tspan);

% ------------------------------
% Matrix computations
% ------------------------------

% 1) DFT and iDFT Matrices (as well as no mean matrices)
DFT = 1 / Nx * getDFT(Nx);
iDFT = getiDFT(Nx);

% 2) True matrix A not in Fourier
O = zeros(Nx, Nx);
I = eye(Nx);
G = Matrix_A_dz(Nx, dx, d);
[~, C] = GetC(abs(xspan - pi) <= a);
B = [O, O; O, -gain * (C' * C)];
A = [O, -g * I; G, O];

% Vector with list of all the positive frequencies
pos_frequences = frequences(2:ceil((length(frequences)-1)/2)+1);

% Matrix that deletes the mean
Pi_nm = iDFT * diag([0; ones(Nx-1, 1)]) * DFT;

% Matrix that deletes the mean and the high frequencies
Pi_lf = iDFT * diag([0; (pos_frequences <= N_ini_fq)'; (pos_frequences <= N_ini_fq)']) * DFT;
Pi_hf = iDFT * diag([0; (pos_frequences > N_ini_fq)'; (pos_frequences > N_ini_fq)']) * DFT;

B_tilde1 = [O, O; O, - gain * Pi_nm * (C' * C)];
B_tilde2 = [O, O; O, - gain * Pi_lf * (C' * C)];

M = A + B;
M_tilde1 = A + B;
M_tilde2 = A + B;

% Function handle for matrix-vector multiplication
func_M = @(y, t) M * y;
func_M_tilde1 = @(y, t) M * y;
func_M_tilde2 = @(y, t) M * y;

% Plot the sparsity pattern of M
figure;
spy(M);
title('Sparsity Pattern of M');

% Compute the eigenvalues and filter out those with positive real parts
eigval_M = eig(M);
eigval_M_pos = eigval_M(real(eigval_M) > 0);

% Initial conditions
eta0 = sin(xspan) + cos(2*xspan);
phi0 = iDFT * ((1j ./ g .* (DFT * eta0')) .* sqrt(g .* abs(frequences)' .* tanh(d * abs(frequences)')));
X0 = [phi0; eta0'];

% Plot eta0
figure;
plot(xspan, real(eta0), '--', xspan, imag(eta0), '--');
title('Initial eta0');
legend('Real Part', 'Imaginary Part');

% Plot phi0
figure;
plot(xspan, real(phi0), xspan, imag(phi0));
title('Initial phi0');
legend('Real Part', 'Imaginary Part');

% Solve the ODE using ode45
[t, X_sol_M] = ode45(@(t, y) func_M(y, t), tspan, X0);
[t, X_sol_M_tilde1] = ode45(@(t, y) func_M_tilde1(y, t), tspan, X0);
[t, X_sol_M_tilde2] = ode45(@(t, y) func_M_tilde2(y, t), tspan, X0);

% Calculate the norm of X_sol_EE_A_mix along the second dimension
norm_X_sol_M = vecnorm(X_sol_M, 2, 2);
norm_X_sol_M_tilde1 = vecnorm(X_sol_M_tilde1, 2, 2);
norm_X_sol_M_tilde2 = vecnorm(X_sol_M_tilde2, 2, 2);


% Plot the norm using a semi-logarithmic scale
figure;
semilogy(tspan, norm_X_sol_M,...
        tspan, norm_X_sol_M_tilde1,...
        tspan, norm_X_sol_M_tilde2);
title('Norm of Solution Over Time');
xlabel('Time');
ylabel('Norm');


%===========================================================================================================
%===========================================================================================================

% Supporting functions
function DFT = getDFT(N)
    w_N = exp(-2i * pi / N);
    [rows, cols] = ndgrid(0:N-1, 0:N-1);
    DFT = w_N .^ (rows .* cols);
end

function iDFT = getiDFT(N)
    w_N = exp(2i * pi / N);
    [rows, cols] = ndgrid(0:N-1, 0:N-1);
    iDFT = w_N .^ (rows .* cols);
end

function A_dz = Matrix_A_dz(Nx, dx, d)
    k = 2 * pi * fftfreq(Nx, dx);
    kernel = real(ifft(abs(k) .* tanh(d * abs(k)))) * Nx;
    A_dz = zeros(Nx, Nx);
    for j = 1:Nx
        A_dz(j, :) = (1 / Nx) * circshift(kernel, [0, j-1]);
    end
end

function k = fftfreq(n,d)
    % f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
    % f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    if mod(n,2)
        k = [[0:(n-1)/2],[-(n-1)/2:-1]]./(d*n);
    else
        k = [[0:n/2-1],[-n/2:-1]]./(d*n);
    end
end

function [nb_obs, C] = GetC(obs_map)
    C_flat = obs_map(:);
    nb_obs = sum(C_flat ~= 0);
    C = zeros(nb_obs, length(C_flat));
    j = 1;
    for p = 1:length(C_flat)
        if C_flat(p) ~= 0
            C(j, p) = C_flat(p);
            j = j + 1;
        end
    end
end