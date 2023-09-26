clear all
close all

fontSize = 16;
all_bf = 0;
film = 1;

%%

% parameters to tune :
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 128;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.5;       % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 0;        % start time
T        = 10;         % time window
N_s      = 1;         % Number of wave frequency in the solution wave
Nbf      = 3;

gain     = 15;     % gamma parameter
p        = 16;     % #CPU

% gain 15 = (1e-10 en T=3)
%           (1e-5  en T=1.5)
% gain 8  = (1e-10 en T=5.5)
%           (1e-5  en T=2.75)
% gain 5  = (1e-10 en T=9)
%           (1e-5  en T=4.5)
% gain 3  = (1e-10 en T=14)
%           (1e-5  en T=7)
% gain 2  = (1e-10 en T=22)
%           (1e-5  en T=11)

%   --------------------
%-- Space discretization
%   --------------------

dx    = Lx/(Nx - 1);        % space steps size
L     = Lx + dx;            % L, for the frequency of the solution
xspan = [x0:dx:Lx];         % space grid   
A_dz  = Matrix_A_dz(Nx,dx); % kernel matrix


%   --------------------
%-- Time discretization
%   --------------------
T_end   = T_start + T;    % end of time interval
tspan = T_start:dt:T_end; %
Nt    = length(tspan);

%   --------------------
%-- Matrices
%   --------------------

A  = [zeros(Nx), -g.*eye(Nx); A_dz, zeros(Nx)];
Pn = ProjBF(Nx,dx,Nbf);
frequences = fftfreq(Nx,dx);
PN = [Pn, zeros(Nx); zeros(Nx), Pn];

%   --------------------
%-- Solution
%   --------------------

a_s     = ones(1,N_s);    % frequencies
kw_s    = 2*pi*[1:N_s]/L; % periods
omega_s = sqrt(kw_s.*g);  % time shift (to be accurate on a physics level)

% solution
eta_s = @(t) sum(a_s'.*sin(omega_s'.*t - kw_s'*xspan))';
phi_s = @(t) sum(a_s'.*(g.*diag(1./omega_s)*cos(omega_s'.*t - kw_s'.*xspan)))';
U_s   = @(t) [phi_s(t); eta_s(t)];

% derivative of the solution
eta_s_dt = @(t) sum(a_s'.*(omega_s'.*cos(omega_s'.*t - kw_s'*xspan)))';
phi_s_dt = @(t) sum(-g.*a_s'.*(sin(omega_s'.*t - kw_s'*xspan)))';
U_dt     = @(t) [phi_s_dt(t); eta_s_dt(t)];


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