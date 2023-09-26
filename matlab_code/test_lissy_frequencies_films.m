clear all
close all

fontSize = 16;
all_bf = 0;
film = 0;

%%

% parameters to tune :
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 128;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.5;         % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 15;        % start time
T        = 10;         % time window
N_s      = 2;         % Number of wave frequency in the solution wave
Nbf      = 2;

gain     = 100;     % gamma parameter
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
G_zero = @(t) zeros(2*Nx,1);
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

M_bf_1       = A - PN * Lmat * C;
Gobs_s_bf_1  = @(t) PN * Lmat * C * U_s(t);

if all_bf
M_bf_2       = PN * A - Lmat * C * PN;
Gobs_s_bf_2  =  @(t) Lmat * C * PN * U_s(t);

M_bf_3       = PN * A * PN - Lmat * C * PN;
Gobs_s_bf_3  =  @(t) Lmat * C * PN * U_s(t);
end

%   ---------------
%-- Theorical decay
%   ---------------
[V_m,E_m] = eig(full(M));
cond_m    = cond(V_m);
zz        = abs(diag(E_m))>1e-10;
E_m       = diag(E_m);
E_m       = E_m(zz);
mu        = min(abs(E_m));

error_th  = cond_m*exp(-(tspan-T_start).*mu);

%   -----------------------
%-- Solver state & observer
%   -----------------------
fprintf('...solving state equation RK4... \n')
[~,solve_state_RK] = odeRK4_inhom_ufunc(sparse(A),G_zero,tspan,U_s(T_start));
fprintf('...solving observer equation RK4... \n')
[~,solve_obs_RK] = odeRK4_inhom_ufunc(sparse(M),Gobs_s,tspan,zeros(2*Nx,1));

%   ------------------------
%-- Solver observers w/ proj
%   ------------------------
fprintf('...solving observer bf1 equation RK4... \n')
[~,solve_obs_RK_bf1] = odeRK4_inhom_ufunc(sparse(M_bf_1),Gobs_s_bf_1,tspan,zeros(2*Nx,1));

if all_bf
fprintf('...solving observer bf2 equation RK4... \n')
[~,solve_obs_RK_bf2] = odeRK4_inhom_ufunc(sparse(M_bf_2),Gobs_s_bf_2,tspan,zeros(2*Nx,1));
fprintf('...solving observer bf3 equation RK4... \n')
[~,solve_obs_RK_bf3] = odeRK4_inhom_ufunc(sparse(M_bf_3),Gobs_s_bf_3,tspan,zeros(2*Nx,1));
end

%   -----------------
%-- Error computation
%   -----------------
error_state_RK  = solve_state_RK - Ugrid_s;
error_obs_RK    = solve_obs_RK - Ugrid_s;
error_obs_RK_bf = solve_obs_RK_bf1 - Ugrid_s;

norm_error_state_RK = vecnorm(error_state_RK);
norm_error_obs_RK = vecnorm(error_obs_RK);
norm_error_obs_RK_bf = vecnorm(error_obs_RK_bf);

if all_bf
error_obs_RK_bf2 = solve_obs_RK_bf2 - Ugrid_s;
error_obs_RK_bf3 = solve_obs_RK_bf3 - Ugrid_s;

norm_error_obs_RK_bf2 = vecnorm(error_obs_RK_bf2);
norm_error_obs_RK_bf3 = vecnorm(error_obs_RK_bf3);
end

%   -----------------
%-- Convergence plots
%   -----------------

figure(1)
semilogy(tspan,norm_error_state_RK,tspan,norm_error_obs_RK,tspan,norm_error_obs_RK_bf,tspan,error_th,'-k')
legend('error $x$ RK4','error $\hat{x}$ (RK4)','error $\hat{x}_{BF}$', 'Interpreter', 'latex','FontSize', fontSize)
xlabel('time $t$', 'Interpreter', 'latex','FontSize', fontSize)

if all_bf
figure(1)
semilogy(tspan,norm_error_state_RK,'-k',tspan,norm_error_obs_RK,'-or',tspan,norm_error_obs_RK_bf,'-+b',tspan,norm_error_obs_RK_bf2,'--c',tspan,norm_error_obs_RK_bf3,'--m')
legend('error $x$ RK4','error $\hat{x}$ (RK4)','error $\hat{x}_{BF1}$','error $\hat{x}_{BF2}$','error $\hat{x}_{BF3}$', 'Interpreter', 'latex','FontSize', fontSize)
xlabel('time $t$', 'Interpreter', 'latex','FontSize', fontSize)

figure(2)
semilogy(tspan,vecnorm(solve_obs_RK_bf1 - solve_obs_RK_bf2),'--r',tspan,vecnorm(solve_obs_RK_bf2 - solve_obs_RK_bf3),'--b',tspan,vecnorm(solve_obs_RK_bf3 - solve_obs_RK_bf1),'--k')
legend('1-2','2-3','3-1', 'Interpreter', 'latex','FontSize', fontSize)
xlabel('time $t$', 'Interpreter', 'latex','FontSize', fontSize)
title('error between observers bf1 , bf2 and bf3', 'Interpreter', 'latex','FontSize', fontSize)
end
%   --------------
%-- Plots one step
%   --------------

nb_frames = 50;
frame_list = 1:floor(length(tspan)/nb_frames):length(tspan);

i = length(frame_list);

frame = frame_list(i);

figure(3)

x0=1;
y0=1;
width=1600;
height=1200;
set(gcf,'position',[x0,y0,width,height])

% convergence

subplot(2,4,[1 5])
semilogy(tspan(frame_list(1:i)),norm_error_obs_RK(frame_list(1:i)),'-o',tspan(frame_list(1:i)),norm_error_obs_RK_bf(frame_list(1:i)),'-x',tspan(frame_list),error_th(frame_list),'-k')
xlabel('time $t$', 'Interpreter', 'latex','FontSize', fontSize)
ylim([1e-12 1e2])
grid on

% phi

subplot(2,4,2)
plot(frequences, abs(fft(error_obs_RK(1:Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,200])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,3)
plot(frequences, abs(fft(error_obs_RK(1:Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,4)
semilogy(frequences, abs(fft(error_obs_RK(1:Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-12,50])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

% eta

subplot(2,4,6)
plot(frequences, abs(fft(error_obs_RK(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,200])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,7)
plot(frequences, abs(fft(error_obs_RK(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,8)
semilogy(frequences, abs(fft(error_obs_RK(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-12,200])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

pause(0.2)

%%

%   -----------
%-- Plots films
%   -----------

if film

nb_frames = 50;
frame_list = 1:floor(length(tspan)/nb_frames):length(tspan);

for i = 1:length(frame_list)

frame = frame_list(i);
    
figure(4)

x0=1;
y0=1;
width=1800;
height=1200;
set(gcf,'position',[x0,y0,width,height])

% convergence

subplot(2,4,[1 5])
semilogy(tspan(frame_list(1:i)),norm_error_obs_RK(frame_list(1:i)),'-o',tspan(frame_list(1:i)),norm_error_obs_RK_bf(frame_list(1:i)),'-x',tspan(frame_list),error_th(frame_list),'-k')
xlabel('time $t$', 'Interpreter', 'latex','FontSize', fontSize)
ylim([1e-12 1e2])
grid on

% phi

subplot(2,4,2)
plot(frequences, abs(fft(error_obs_RK(1:Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,50])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,3)
plot(frequences, abs(fft(error_obs_RK(1:Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,4)
semilogy(frequences, abs(fft(error_obs_RK(1:Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-12,50])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

% eta

subplot(2,4,6)
plot(frequences, abs(fft(error_obs_RK(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,50])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,7)
plot(frequences, abs(fft(error_obs_RK(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,8)
semilogy(frequences, abs(fft(error_obs_RK(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(error_obs_RK_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-12,50])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$\hat{x} - x$ (RK4)','$\hat{x}_{BF} - x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

pause(0.2)

end


end

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