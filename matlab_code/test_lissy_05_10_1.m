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
T        = 50;        % time window
Nif      = 2;         % Number of wave frequency in the solution wave
Nbf      = 2;

gain     = 2*floor(exp(2*Nbf+1));     % gamma parameter
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

%   -----------------------
%-- True solution
%   -----------------------
a     = ones(1,Nif);    % frequencies
kw    = 2*pi*[1:Nif]/L; % periods
omega = sqrt(kw.*g);  % time shift (to be accurate on a physics level)

% solution
eta = @(t) sum(a'.*sin(omega'.*t - kw'*xspan))';
phi = @(t) sum(a'.*(g.*diag(1./omega)*cos(omega'.*t - kw'.*xspan)))';
U   = @(t) [phi(t); eta(t)];

% derivative of the solution
eta_dt = @(t) sum(a'.*(omega'.*cos(omega'.*t - kw'*xspan)))';
phi_dt = @(t) sum(-g.*a'.*(sin(omega'.*t - kw'*xspan)))';
U_dt     = @(t) [phi_dt(t); eta_dt(t)];

%   -----------------------
%-- True solution grid
%   -----------------------
U_mat = zeros(2*Nx,Nt);
for t=1:Nt
    U_mat(:,t) = U(tspan(t));
end

%   -----------------------
%-- Initial conditions
%   -----------------------
U0 = U(0);
O0 = zeros(2*Nx,1);
E0 = U0 - O0;

%   -----------------------
%-- Observer setting
%   -----------------------
obs_map    = abs(xspan) <= size_obs;
%obs_map    = abs(x - Lx/2) > size_obs*Lx/2;
%obs_map    = ones(1,Nx);
[m_obs,Cc] = GetC(obs_map);
C          = [zeros(m_obs,Nx) Cc];
%C          = [zeros(Nx), eye(Nx)];
%Lmat       = gain * C';
Lmat       = C';
M          = A - Lmat * C;

% low frequencies
M_bf       = A - PN * Lmat * C;

%   -----------------------
%-- Observer source term
%   -----------------------
Gobs     = @(t) Lmat * C * U(t);

Gobs_mat = zeros(2*Nx,Nt);
for t=1:Nt
    Gobs_mat(:,t) = Lmat * C * U_mat(:,t);
end

%low frequencies
Gobs_bf = @(t) PN * Lmat * C * U(t);
Gobs_bf_mat = zeros(2*Nx,Nt);
for t=1:Nt
    Gobs_bf_mat(:,t) = PN * Lmat * C * U_mat(:,t);
end

%   -----------------------
%-- Zero source term
%   -----------------------
G_zero = @(t) zeros(2*Nx,1);
G_zero_mat = zeros(2*Nx,Nt);

%   -----------------------
%-- Solver
%   -----------------------
tic

fprintf('...solving state equation RK4... \n')
[~,solve_state] = odeRK4_inhom_ufunc(sparse(A),G_zero,tspan,U0);
fprintf('...solving observer equation RK4... \n')
[~,solve_obs] = odeRK4_inhom_ufunc(sparse(M),Gobs,tspan,O0);
fprintf('...solving observer equation BF RK4... \n')
[~,solve_obs_bf] = odeRK4_inhom_ufunc(sparse(M_bf),Gobs_bf,tspan,O0);

cpu = toc;

fprintf('\n == time all solve = %d seconds == \n', cpu)

tic

fprintf('...solving state equation RK4 [matrix source term]... \n')
[~,solve_state_mat] = odeRK4_inhom_umat(sparse(A),G_zero_mat,tspan,U0);
fprintf('...solving observer equation RK4 [matrix source term]... \n')
[~,solve_obs_mat] = odeRK4_inhom_umat(sparse(M),Gobs_mat,tspan,O0);
fprintf('...solving observer equation BF RK4 [matrix source term]... \n')
[~,solve_obs_bf_mat] = odeRK4_inhom_umat(sparse(M_bf),Gobs_bf_mat,tspan,O0);

cpu = toc;

fprintf('\n == time all solve = %d seconds == \n', cpu)

%   -----------------------
%-- Errors
%   -----------------------
error_state  = vecnorm(solve_state - U_mat);
error_obs = vecnorm(solve_obs - U_mat);
error_obs_bf = vecnorm(solve_obs_bf - U_mat);

error_state_mat  = vecnorm(solve_state_mat - U_mat);
error_obs_mat = vecnorm(solve_obs_mat - U_mat);
error_obs_bf_mat = vecnorm(solve_obs_bf_mat - U_mat);


%% PLOTS

%   -----------------------
%-- Initial condition
%   -----------------------

phi_0 = phi(0);
eta_0 = eta(0);

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
    tspan(points_list),error_obs_bf(points_list),'-r',...
    tspan(points_list),error_obs(points_list),'-m',...
    tspan(points_list),error_state(points_list),'-b',...
    tspan(points_list),error_obs_bf_mat(points_list),'xr',...
    tspan(points_list),error_obs_mat(points_list),'xm',...
    tspan(points_list),error_state_mat(points_list),'xb')
legend('$\|\hat{x}_{BF} - x\|$','$\|\hat{x} - x\|$','$\|\x - x_{RK4}\|$','error obs bf mat','error obs mat','error state mat', 'Interpreter', 'latex','FontSize', fontSize)
title('convergence plots, $U_0 = $ physical values', 'Interpreter', 'latex','FontSize', fontSize)
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

function A_dz = Matrix_B_dz(Nx,dx)
    k      = 2*pi*fftfreq(Nx, dx);
    kernel = real(ifft(abs(k)^2))*Nx;
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