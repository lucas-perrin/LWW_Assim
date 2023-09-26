clear all
close all

fontSize = 16;

%%

% parameters to tune :
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 128;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.9;         % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 15;        % start time
T        = 3;         % time window
N_s      = 2;         % Number of wave frequency in the solution wave
Nbf      = 2;

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

% source term
G = @(t) zeros(2*Nx,1);
%G = @(t) U_dt(t) - A * U_s(t);

%--
Ugrid_s = zeros(2*Nx,Nt);
for t=1:Nt
    Ugrid_s(:,t) = U_s(tspan(t));
end

%   ----------------
%-- Observer setting
%   ----------------


obs_map    = abs(xspan) <= size_obs;
%obs_map    = abs(x - Lx/2) > size_obs*Lx/2;
%obs_map    = ones(1,Nx);
[m_obs,Cc] = GetC(obs_map);
C          = [zeros(m_obs,Nx) Cc];
%C          = [zeros(Nx), eye(Nx)];
Lmat       = gain * C';
M          = A - Lmat * C;

Y_s        = @(t) C * U_s(t);
Gobs_s     = @(t) Lmat * C * U_s(t);


% low frequencies
M_bf_1       = PN * A - Lmat * C * PN;
Gobs_s_bf_1  = @(t) Lmat * C * PN * U_s(t);

M_bf_2       = A * PN - Lmat * C * PN;
Gobs_s_bf_2  =  @(t) Lmat * C * PN * U_s(t);

M_bf_3       = PN * A * PN - Lmat * C * PN;
Gobs_s_bf_3  =  @(t) Lmat * C * PN * U_s(t);


fprintf("computing U state \n")
[~,Ustate_s] = odeRK4_inhom_ufunc(sparse(A),G,tspan,U_s(T_start));
fprintf("computing U obs \n")
[~,Uobs_s]  = odeRK4_inhom_ufunc(sparse(M),Gobs_s,tspan,U_s(0));
fprintf("computing U obs BF 1 \n")
[~,Uobs_s_bf_1]  = odeRK4_inhom_ufunc(sparse(M_bf_1),Gobs_s_bf_1,tspan,U_s(0));
fprintf("computing U obs BF 2 \n")
[~,Uobs_s_bf_2]  = odeRK4_inhom_ufunc(sparse(M_bf_2),Gobs_s_bf_2,tspan,U_s(0));
fprintf("computing U obs BF 3 \n")
[~,Uobs_s_bf_3]  = odeRK4_inhom_ufunc(sparse(M_bf_3),Gobs_s_bf_3,tspan,U_s(0));

error_state_s  = vecnorm(Ustate_s - Ugrid_s);
error_obs_s    = vecnorm(Uobs_s - Ugrid_s);
error_obs_s_bf_1 = vecnorm(PN*(Uobs_s_bf_1 - Ugrid_s));
error_obs_s_bf_2 = vecnorm(PN*(Uobs_s_bf_2 - Ugrid_s));
error_obs_s_bf_3 = vecnorm(PN*(Uobs_s_bf_3 - Ugrid_s));


[V_m,E_m] = eig(full(M));
cond_m    = cond(V_m);
zz        = abs(diag(E_m))>1e-10;
E_m       = diag(E_m);
E_m       = E_m(zz);
mu        = min(abs(E_m));
error_th  = @(t) cond_m*exp(-(t-T_start).*mu);

error_vec = abs(Uobs_s(1:Nx,:) - Ugrid_s(1:Nx,:));

figure(1)
subplot(1,2,1)
semilogy(tspan,error_obs_s,'r',tspan,error_obs_s_bf_1,'m',tspan,error_obs_s_bf_2,'m--',tspan,error_obs_s_bf_3,'mx',tspan,error_state_s,'b',tspan,error_th(tspan),'--k')
legend('error observer','error observer bf','error observer bf proj','error discretization','theorical error', 'Interpreter', 'latex')
ylim([1e-12, 1e1])
xlabel("time $t$", 'Interpreter', 'latex','FontSize', fontSize)
ylabel("$|| \epsilon(t) ||$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
title(['(LWWE) norm error convergence plot, $N_x = 128$, $\Delta_t = 5 \cdot 10^{-4}$, $T_0 = 15$'], 'Interpreter', 'latex','FontSize', fontSize)

subplot(1,2,2)
surf(tspan,xspan,error_vec)
set(gca, 'ZScale', 'log','ColorScale','log')
shading interp
view(2)
colorbar
caxis([1e-12 1e-1])
xlabel("time $t$", 'Interpreter', 'latex','FontSize', fontSize)
ylabel("$x$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
title(['(LWWE) space error convergence plot, $N_x = 128$, $\Delta_t = 5 \cdot 10^{-4}$, $T_0 = 15$'], 'Interpreter', 'latex','FontSize', fontSize)


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