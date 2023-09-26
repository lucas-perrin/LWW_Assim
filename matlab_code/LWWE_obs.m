clear all
close all

fontSize = 16;
test = 1;
eff_test = 0;

%% > ONLY Munch source term

% Observer parameters

x0       = 0;           % start of space domain
Nx       = 128;         % Number of points in space domain
L        = 1;           % length of space domain
dt       = 1e-3;        % timestep
size_obs = 1;           % value between 0 and 1
g        = 9.81;        % gravity constant
T_start  = 0;          % start time
T        = 5;           % time window
T_end    = T_start + T; % end time
N_s      = 10;          % Number of wave frequency in the solution wave

p        = 16;  % #CPU
gain     = 100;  % gamma parameter

% Good parameters :

% 



% space discretization
xspan = linspace(x0, x0+L, Nx); % space grid
dx    = xspan(2) -xspan(1);      % step in space
dxsp  = L/(Nx-1);
xsp   = 0:dxsp:L;


% time discretization
tspan = T_start:dt:T_end;
Nt    = length(tspan);

% get the matrices
A_dz = Matrix_A_dz(Nx,dxsp);

A  = [zeros(Nx), -g.*eye(Nx); A_dz, zeros(Nx)];

% solution
N_s     = 10;
a_s     = ones(1,N_s).*[N_s:-1:1]./(N_s^2)./4;
kw_s    = 2*pi*[1:N_s]/L;
omega_s = sqrt(kw_s.*g);

eta_s   = @(t) sum(a_s'.*sin(omega_s'.*t - kw_s'*xsp));
phi0_s  = @(t) sum(a_s'.*(g.*diag(1./omega_s)*cos(omega_s'.*t - kw_s'.*xsp)));

U_s     = @(t) [reshape(phi0_s(t),Nx,1);reshape(eta_s(t),Nx,1)];

% G     = @(t) zeros(2*Nx,1);

eta_s_dt  = @(t) sum(a_s'.*(omega_s'.*cos(omega_s'.*t - kw_s'*xsp)));
phi0_s_dt = @(t) sum(-g.*a_s'.*(sin(omega_s'.*t - kw_s'*xsp)));
U_dt      = @(t) [reshape(phi0_s_dt(t),Nx,1);reshape(eta_s_dt(t),Nx,1)];
G         = @(t) U_dt(t) - A*U_s(t);

Ugrid_s = zeros(2*Nx,Nt);
for t=1:Nt
    Ugrid_s(:,t) = U_s(tspan(t));
end

U0 = U_s(T_start);

% Observer setting
U0obs      = zeros(2*Nx,1);
%obs_map    = ones(1,Nx);
%[m_obs,Cc] = GetC(obs_map);
%C          = [zeros(m_obs,Nx) Cc];
C          = [zeros(Nx),eye(Nx)];
Lmat       = gain * C';
M          = A - Lmat*C;

Y_s        = @(t) C * U_s(t);
Gobs_s     = @(t) G(t) + Lmat*Y_s(t);

% State
if test
[~,Ustate_s] = odeRK4_inhom_ufunc(sparse(A),G,tspan,U_s(T_start));
end
% Observer
if test
[~,Uobs_s]   = odeRK4_inhom_ufunc(sparse(M),Gobs_s,tspan,zeros(2*Nx,1));
end

if test
error_state_s = vecnorm(Ustate_s - Ugrid_s);
error_obs_s   = vecnorm(Uobs_s - Ugrid_s);
error_vec = abs(Uobs_s(1:Nx,:) - Ugrid_s(1:Nx,:));
end

% theorical decay
if test
[V_m,E_m] = eig(full(M));
cond_m    = cond(V_m);
zz        = abs(diag(E_m))>1e-10;
E_m       = diag(E_m);
E_m       = E_m(zz);
mu        = min(abs(E_m));

error_th  = @(t) cond_m*exp(-(t-T_start).*mu);
end


%% PinT scheme & comparaison

if eff_test
%T_list  = [0.05,0.1,0.5,1,2,5,10,15];
T_list  = 10.^[-1.5:0.4:1.3];
T_list  = 10.^[-2:0.4:1];
toler   = 1e-5;

Results = zeros(1,2);

%----- Serial scheme

fprintf("SERIAL SCHEME \n")

debut = cputime();

[tspan_serial,X_hat_max] = odeRK4_inhom_ufunc_Stop(sparse(M),Gobs_s,tspan,U0obs,toler,U);

time_serial = cputime() - debut;

for ti = 1:length(T_list)
    
T        = T_list(ti);
L_window = floor(Tf/T);
dim_int  = floor(nt/L_window);

fprintf("==== T = %d || gain = %d ==== \n \n",T,gain)

%----- Serial scheme time

fprintf("t_end serial = %d \n",tspan_serial(end))

%----- Choix facile

fprintf("PARALLEL SCHEME \n")

x0obsp = U0obs;

time_para = 0;

%----- Iteration on the windows
for l = 1:L_window

    fprintf("Window number = %d / %d \n",l,L_window)

    if l == 1
        W_l = tspan(1:dim_int);
    else
        if l < L_window
        W_l = tspan((l-1)*dim_int:l*dim_int);
        else
            W_l = tspan((l-1)*dim_int:end);
        end
    end
    
    %----- ParaExp Solver
    
    [X_hat_para_end_l,time_l] = ParExp_max_S(p,W_l,sparse(M),Gobs_s,x0obsp);
    time_para = time_para + time_l;
    
    if norm(X_hat_para_end_l - U(W_l(end))) < toler
        tspan_para = tspan(1:(l<L_window)*l*dim_int + (l==L_window)*nt);
        break
    else
        tspan_para = tspan;
    end

    x0obsp = X_hat_para_end_l;

end

fprintf('T : %d, gain : %d, size_obs : %d, p : %d \n',T, gain, size_obs, p)

fprintf('efficiency : %d \n',(time_serial/time_para))

fprintf('-cpu time serial : %d \n-cpu time parall : %d \n',time_serial, time_para)
fprintf('*stop time serial : %d \n*stop time parall : %d \n',tspan_serial(end),tspan_para(end))

Results(ti) = (time_serial/time_para);

end

disp(T_list);disp(Results);
end

%%

if test
figure(4)
subplot(1,2,1)
semilogy(tspan,error_obs_s,'r',tspan,error_state_s,'b',tspan,error_th(tspan),'--k')
legend('error observer','error discretization','theorical error', 'Interpreter', 'latex')
ylim([1e-12, 1e1])
xlabel("time $t$", 'Interpreter', 'latex','FontSize', fontSize)
ylabel("$|| \epsilon(t) ||$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
title(['(LWWE) norm error convergence plot, $N_x = 128$, $\Delta_t = 10^{-5}$, $T_0 = 15$'], 'Interpreter', 'latex','FontSize', fontSize)

subplot(1,2,2)
surf(tspan,xspan,error_vec)
set(gca, 'ZScale', 'log','ColorScale','log')
shading interp
view(2)
colorbar
caxis([1e-12 1e-1])
xlabel("time $t$", 'Interpreter', 'latex','FontSize', fontSize)
ylabel("$x$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
title(['(LWWE) space error convergence plot, $N_x = 128$, $\Delta_t = 10^{-5}$, $T_0 = 15$'], 'Interpreter', 'latex','FontSize', fontSize)
end

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