clear all
close all

fontSize = 16;

%%

% Observer parameters

x0       = 0;           % start of space domain
Nx       = 128;         % Number of points in space domain
Lx       = 1;           % length of space domain
dt       = 1e-3;        % timestep
size_obs = 1;           % value between 0 and 1
g        = 9.81;        % gravity constant
T_start  = 15;          % start time
T        = 5;           % time window
T_end    = T_start + T; % end time
N_s      = 30;          % Number of wave frequency in the solution wave

p        = 16;  % #CPU
gain     = 20;  % gamma parameter

% space discretization
xspan = linspace(x0, x0+Lx, Nx); % space grid
dx    = xspan(2)-xspan(1);      % step in space
dxsp  = Lx/(Nx - 1);
xsp   = 0:dxsp:Lx;


% time discretization
tspan = T_start:dt:T_end;
Nt    = length(tspan);

% get the matrices
A_dz = Matrix_A_dz(Nx,dxsp);

Lap1D = ((Nx+1)^2).*tridiag(1,-2,1,Nx);

A  = [zeros(Nx), -g.*eye(Nx); A_dz, zeros(Nx)];

A2 = [zeros(Nx), Lap1D-g.*eye(Nx); A_dz, zeros(Nx)];

% define the solution

%a_s     = ones(1,N_s);
a_s     = ones(1,N_s).*[N_s:-1:1]./(N_s^2)./4;
kw_s    = 2*pi*[1:N_s]/(Lx + dx);
omega_s = sqrt(kw_s.*g);

eta_s   = @(t) sum(a_s'.*sin(omega_s'.*t - kw_s'*xsp))';
phi0_s  = @(t) sum(a_s'.*(g.*diag(1./omega_s)*cos(omega_s'.*t - kw_s'.*xsp)))';

U_s     = @(t) [phi0_s(t); eta_s(t)];

eta_s_dt  = @(t) sum(a_s'.*(omega_s'.*cos(omega_s'.*t - kw_s'*xsp)))';
phi0_s_dt = @(t) sum(-g.*a_s'.*(sin(omega_s'.*t - kw_s'*xsp)))';

U_s_dt    = @(t) [phi0_s_dt(t); eta_s_dt(t)];

% source term

%G     = @(t) zeros(2*Nx,1);
G     = @(t) U_s_dt(t) - A*U_s(t);

Ugrid_s = zeros(2*Nx,Nt);
for t=1:Nt
    Ugrid_s(:,t) = U_s(tspan(t));
end

figure(1)
subplot(2,1,1)
plot(xspan,phi0_s(T_start))
xlabel('$|x-x_0|/L$','Interpreter', 'latex','FontSize', fontSize)
title('$\phi_s(T_{start})$','Interpreter', 'latex','FontSize', fontSize)
subplot(2,1,2)
plot(xspan,eta_s(T_start))
xlabel('$|x-x_0|/L$','Interpreter', 'latex','FontSize', fontSize)
title('$\eta_s(T_{start})$','Interpreter', 'latex','FontSize', fontSize)

% State run with RK4

[~,Ustate_s] = odeRK4_inhom_ufunc(sparse(A),G,tspan,U_s(T_start));

phi0state_s = Ustate_s(1:Nx,:);
etastate_s = Ustate_s(Nx+1:2*Nx,:);

figure(2)
subplot(2,1,1)
plot(xspan,phi0_s(T_end),xspan,phi0state_s(:,end),'--r')
xlabel('$|x-x_0|/L$','Interpreter', 'latex','FontSize', fontSize)
title('$\phi_s(T_{end})$','Interpreter', 'latex','FontSize', fontSize)
subplot(2,1,2)
plot(xspan,eta_s(T_end),xspan,etastate_s(:,end),'--r')
xlabel('$|x-x_0|/L$','Interpreter', 'latex','FontSize', fontSize)
title('$\eta_s(T_{end})$','Interpreter', 'latex','FontSize', fontSize)

% Observer setting

U0obs      = zeros(2*Nx,1);
C          = [zeros(Nx),eye(Nx)];
Lmat       = gain * C';
M          = A - Lmat * C;
Y_s        = @(t) C * U_s(t);
Gobs_s     = @(t) G(t) + Lmat * Y_s(t);

[~,Uobs_s]   = odeRK4_inhom_ufunc(sparse(M),Gobs_s,tspan,zeros(2*Nx,1));

phi0obs_s = Uobs_s(1:Nx,:);
etaobs_s = Uobs_s(Nx+1:2*Nx,:);

figure(3)
subplot(2,1,1)
plot(xspan,phi0_s(T_end),xspan,phi0obs_s(:,end),'--r')
xlabel('$|x-x_0|/L$','Interpreter', 'latex','FontSize', fontSize)
title('$\phi_s(T_{end})$','Interpreter', 'latex','FontSize', fontSize)
subplot(2,1,2)
plot(xspan,eta_s(T_end),xspan,etaobs_s(:,end),'--r')
xlabel('$|x-x_0|/L$','Interpreter', 'latex','FontSize', fontSize)
title('$\eta_s(T_{end})$','Interpreter', 'latex','FontSize', fontSize)

% Observer error analysis

error_obs_s = vecnorm(Uobs_s - Ugrid_s);
error_state_s = vecnorm(Ustate_s - Ugrid_s);
error_vec = abs(Uobs_s(1:Nx,:) - Ugrid_s(1:Nx,:));

figure(4)
semilogy(tspan,error_obs_s,tspan,error_state_s)
grid on

figure(5)
surf(tspan,xspan,error_vec)
set(gca, 'ZScale', 'log','ColorScale','log')
shading interp
view(2)
colorbar
caxis([1e-12 1e-1])
xlabel("time $t$", 'Interpreter', 'latex','FontSize', fontSize)
ylabel("$x$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
title(['(LWWE) space error convergence plot, $N_x = 128$, $\Delta_t = 10^{-3}$, $T_0 = 15$'], 'Interpreter', 'latex','FontSize', fontSize)

%% ----- functions

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

function [X_hat_para_end,time] = ParExp_max_S(p,tspan,M,G,xh0)

mx              = length(xh0);
nt              = length(tspan);
size_int        = fix(nt/p);

X_hat_para      = zeros(mx,nt);
X_hat_para(:,1) = xh0;

ini_zero        = zeros(mx,1);

xh0p            = zeros(mx,p);
xh0p(:,1)       = xh0;

time            = 0;

for iter = 1:p
    
%----- Interval definition
    
    deb       = 1*(iter==1) + (iter-1)*size_int*(iter > 1);
    fin       = iter*size_int;
    if iter == p
        fin = length(tspan);
    end
    time_interval = tspan(deb:fin);
    
%----- Type 1
    
    debut = cputime();
    
    [~,X_hat_T1_p] = odeRK4_inhom_ufunc(M,G,time_interval,ini_zero);
    
    temps  = cputime() - debut;
    
    X_hat_para(:,deb:fin) = X_hat_para(:,deb:fin) + X_hat_T1_p;
    
    if iter < p
        xh0p(:,iter+1) = X_hat_T1_p(:,end);
    end
    
    time = time + temps;
    
end

for iter = 1:p
    
%----- Interval definition
    
    deb       = 1*(iter==1) + (iter-1)*size_int*(iter > 1);
    time_interval_till_end_from_zero = tspan(deb:end) - tspan(deb);
    
%----- Type 2

    debut = cputime();

    %[~,X_hat_T2_p] = odeRK4_hom(M,time_interval_till_end,xh0p);
    [X_hat_T2_p_end] = expm(time_interval_till_end_from_zero(end).*M)*xh0p(:,iter);
    
    temps  = cputime() - debut;
    
    X_hat_para(:,end) = X_hat_para(:,end) + X_hat_T2_p_end;
    
    time = time + temps;
    
end

X_hat_para_end = X_hat_para(:,end);

end