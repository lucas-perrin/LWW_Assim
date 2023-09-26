clear all
close all

fontSize = 16;

%% COMPUTATIONS

%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 128;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.9;       % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 0;         % start time
T        = 20;        % time window
Nif      = 2;         % Number of wave frequency in the solution wave
Nbf      = 50;

%gain     = floor(exp(2*Nbf+1));     % gamma parameter
gain     = 1;
p        = 16;     % #CPU

%   -----------------------
%-- Space discretization
%   -----------------------
dx    = Lx/(Nx - 1);        % space steps size
L     = Lx + dx;            % L, for the frequency of the solution
xspan = [x0:dx:Lx];         % space grid   
A_dz  = Matrix_A_dz(Nx,dx); % kernel matrix

%   -----------------------
%-- Time discretization
%   -----------------------
T_end   = T_start + T;    % end of time interval
tspan = T_start:dt:T_end; %
Nt    = length(tspan);

%   -----------------------
%-- Matrices
%   -----------------------
A  = [zeros(Nx), -g.*eye(Nx); A_dz, zeros(Nx)];
e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
Pn   = ProjBF(Nx,dx,Nbf);
Pn_0 = ProjBF_2(Nx,dx,Nbf);
frequences = fftfreq(Nx,dx);
PN = [Pn_0, zeros(Nx); zeros(Nx), Pn_0];

%   --------------------
%-- Initial condition
%   --------------------

e0_phi = 2*xspan - ones(1,Nx);
e0_eta = 2*xspan - ones(1,Nx);

% for i = 1:Nif
%     e0_phi = e0_phi + sin(i*2*pi*xspan/L);
%     e0_eta = e0_eta + sin(i*2*pi*xspan/L);
% end

U0 = [e0_phi' ; e0_eta'];

O0 = zeros(2*Nx,1);
E0 = U0 - O0;

%   -----------------------
%-- Zero source term
%   -----------------------
G_zero = @(t) zeros(2*Nx,1);
G_zero_mat = zeros(2*Nx,Nt);

%   -----------------------
%-- Solver State
%   -----------------------

fprintf('...solving state equation RK4... \n')
[~,solve_state] = odeRK4_inhom_ufunc(sparse(A),G_zero,tspan,U0);

U_mat = solve_state;

%   -----------------------
%-- Observer setting
%   -----------------------
obs_map    = abs(xspan) <= size_obs;
%obs_map    = abs(x - Lx/2) > size_obs*Lx/2;
%obs_map    = ones(1,Nx);
[m_obs,Cc] = GetC(obs_map);
C          = [zeros(m_obs,Nx) Cc];
%C          = [zeros(Nx), eye(Nx)];
Lmat       = gain * C';
%Lmat       = C';
M          = A - Lmat * C;

% low frequencies
M_bf       = A - PN * Lmat * C;

%   -----------------------
%-- Observer source term
%   -----------------------
Gobs_mat = zeros(2*Nx,Nt);
for t=1:Nt
    Gobs_mat(:,t) = Lmat * C * U_mat(:,t);
end

%low frequencies
Gobs_bf_mat = zeros(2*Nx,Nt);
for t=1:Nt
    Gobs_bf_mat(:,t) = PN * Lmat * C * U_mat(:,t);
end

%   -----------------------
%-- Solver
%   -----------------------
tic

fprintf('...solving observer equation [matrix source term]... \n')
[~,solve_obs_mat] = odeRK4_inhom_umat(sparse(M),Gobs_mat,tspan,O0);
fprintf('...solving error equation BF [matrix source term]... \n')
[~,solve_error_mat] = odeRK4_inhom_umat(sparse(M),G_zero_mat,tspan,E0);

fprintf('...solving observer equation BF RK4 [matrix source term]... \n')
[~,solve_obs_bf_mat] = odeRK4_inhom_umat(sparse(M_bf),Gobs_bf_mat,tspan,O0);
fprintf('...solving error equation BF RK4 [matrix source term]... \n')
[~,solve_error_bf_mat] = odeRK4_inhom_umat(sparse(M_bf),G_zero_mat,tspan,E0);

cpu = toc;

fprintf('\n == time all solve = %d seconds == \n \n', cpu)

%   -----------------------
%-- Errors
%   -----------------------
error_obs_mat = vecnorm(solve_obs_mat - U_mat);
error_mat     = vecnorm(solve_error_mat);
error_obs_bf_mat = vecnorm(solve_obs_bf_mat - U_mat);
error_bf_mat     = vecnorm(solve_error_bf_mat);


%% PLOTS

%   -----------------------
%-- Initial condition
%   -----------------------

phi_0 = e0_phi';
eta_0 = e0_eta';

figure(1)
subplot(1,2,1)
plot(xspan,phi_0)
subplot(1,2,2)
plot(xspan,eta_0)

%   -----------------------
%-- Convergence plot
%   -----------------------
nb_points = 100;
points_list = 1:floor(length(tspan)/nb_points):length(tspan);

figure(2)
semilogy(...
    tspan(points_list),error_obs_bf_mat(points_list),'xr',...
    tspan(points_list),error_obs_mat(points_list),'xm',...
    tspan(points_list),error_bf_mat(points_list),'or',...
    tspan(points_list),error_mat(points_list),'om')
legend('$\|\hat{x}_{BF} - x\|$ mat','$\|\hat{x} - x\|$ mat','$\|\varepsilon_{BF}\|$ mat','$\|\varepsilon\|$ mat', 'Interpreter', 'latex','FontSize', fontSize)
title('convergence plots, $U_0 =  [\sin, \sin]$', 'Interpreter', 'latex','FontSize', fontSize)
grid on


%% functions

function k = fftfreq(n,d)
    % f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
    % f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    if mod(n,2)
        k = [[0:(n-1)/2],[-(n-1)/2:-1]]./(d*n);
    else
        k = [[0:n/2-1],[-n/2:-1]]./(d*n);
    end
end

function A_dz = Matrix_A_dz(Nx,dx)
    k      = 2*pi*fftfreq(Nx, dx);
    kernel = real(ifft(abs(k)))*Nx;
    A_dz   = zeros(Nx);
    for i = 0:Nx-1
        A_dz(i+1,:) = (1/Nx)*circshift(kernel,i);
    end
end

function Pn = ProjBF(Nx,dx,Nbf)
    frequences = fftfreq(Nx,dx);
    Low = diag(abs(frequences) < Nbf);
    e_kn_1 = exp(-i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    Pn = real(e_kn_2 * Low * e_kn_1);
end

function Pn = ProjBF_2(Nx,dx,Nbf)
    frequences = fftfreq(Nx,dx);
    Low = diag((abs(frequences) > 0).*(abs(frequences) < Nbf));
    e_kn_1 = exp(-i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    Pn = real(e_kn_2 * Low * e_kn_1);
end

function [nb_obs,C]=GetC(obs_map)

C_flat = obs_map(:)';

nb_obs = sum(C_flat ~= 0);

C = zeros(nb_obs,length(C_flat));

j = 1;

for p = 1:length(C_flat)
    if C_flat(p) ~= 0
        C(j,p) = C_flat(p);
        j = j+1;
    end
end

end