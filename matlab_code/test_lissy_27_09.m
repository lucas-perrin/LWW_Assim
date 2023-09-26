clear all
close all

fontSize = 16;
film = 0;

%%

% 
%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 128;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.5;       % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 0;        % start time
T        = 20;         % time window
Nif      = 1;         % Number of wave frequency in the solution wave
Nbf      = 1;

gain     = floor(sqrt(2*Nbf+1));     % gamma parameter
gain     = 1;     % gamma parameter
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
e_kn_1 = exp(-i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
Pn   = ProjBF(Nx,dx,Nbf);
Pn_0 = ProjBF_2(Nx,dx,Nbf);
frequences = fftfreq(Nx,dx);
PN = [Pn_0, zeros(Nx); zeros(Nx), Pn_0];

%   -----------------------
%-- Initial condition
%   -----------------------
e0_phi = zeros(1,Nx);
e0_eta = zeros(1,Nx);

for i = 1:Nif
    e0_phi = e0_phi + sin(i*2*pi*xspan/L);
    e0_eta = e0_eta + sin(i*2*pi*xspan/L);
end

E0 = [e0_phi' ; e0_eta'];

%   -----------------------
%-- Zero source term
%   -----------------------
G_zero = @(t) zeros(2*Nx,1);

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
M          = A - Lmat * C;

% low frequencies

M_bf       = A - PN * Lmat * C;

%   -----------------------
%-- ...
%   -----------------------
dft_A = [e_kn_1, zeros(Nx); zeros(Nx), e_kn_1] * A;
dft_M = [e_kn_1, zeros(Nx); zeros(Nx), e_kn_1] * M;
dft_M_bf = [e_kn_1, zeros(Nx); zeros(Nx), e_kn_1] * M_bf;

%   -----------------------
%-- Theorical decay
%   -----------------------
[V_m,E_m] = eig(full(M));
cond_m    = cond(V_m);
zz        = abs(diag(E_m))>1e-10;
E_m       = diag(E_m);
E_m       = E_m(zz);
mu        = min(abs(E_m));

error_th  = cond_m*exp(-(tspan-T_start).*mu);

%   -----------------------
%-- Solver state
%   -----------------------
fprintf('...solving error equation RK4... \n')
[~,solve_state] = odeRK4_inhom_ufunc(sparse(A),G_zero,tspan,E0);

%   -----------------------
%-- Solver error
%   -----------------------
fprintf('...solving error equation RK4... \n')
[~,solve_error] = odeRK4_inhom_ufunc(sparse(M),G_zero,tspan,E0);
fprintf('...solving error equation BF RK4... \n')
[~,solve_error_bf] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,E0);

norm_error = vecnorm(solve_error);
norm_error_bf = vecnorm(solve_error_bf);

norm_phi_error = vecnorm(solve_error(1:Nx,:));
norm_eta_error = vecnorm(solve_error(Nx+1:2*Nx,:));

norm_phi_error_bf = vecnorm(solve_error_bf(1:Nx,:));
norm_eta_error_bf = vecnorm(solve_error_bf(Nx+1:2*Nx,:));

%   -----------------------
%-- Plots initial value
%   -----------------------

figure(1)
subplot(2,2,1)
plot(xspan,e0_phi,'-bx')
ylabel("$\phi_0$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
subplot(2,2,2)
stem(frequences,abs(fft(e0_phi)),'bx')
ylabel("$\phi_0$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
subplot(2,2,3)
plot(xspan,e0_eta,'-rx')
ylabel("$\eta_0$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)
subplot(2,2,4)
stem(frequences,abs(fft(e0_eta)),'rx')
ylabel("$\eta_0$", 'Interpreter', 'latex','FontSize', fontSize,'rotation',0)

figure(2)
semilogy(tspan,norm_phi_error,'b',...
    tspan,norm_eta_error,'b',...
    tspan,norm_phi_error_bf,'r',...
    tspan,norm_eta_error_bf,'r')
legend('error phi','error eta','error bf phi','error bf eta')


%   -----------------------
%-- Plots eig matrices
%   -----------------------

eigval_A    = eig(A);
eigval_M    = eig(M);
eigval_M_bf = eig(M_bf);

figure(3)
subplot(3,2,1)
plot(real(eigval_A),imag(eigval_A),'+b')
xlim([-150, 20])
ylim([-80, 80])
title('eigenvalues of $A$', 'Interpreter', 'latex','FontSize', fontSize)
grid on
subplot(3,2,2)
plot(real(eigval_A),imag(eigval_A),'+b')
xlim([-1e-13, 1e-13])
ylim([-80, 80])
title('eigenvalues of $A$', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(3,2,3)
plot(real(eigval_M),imag(eigval_M),'or')
title('eigenvalues of $M$', 'Interpreter', 'latex','FontSize', fontSize)
xlim([-150, 20])
ylim([-80, 80])
grid on
subplot(3,2,4)
plot(real(eigval_M),imag(eigval_M),'or')
title('eigenvalues of $M$', 'Interpreter', 'latex','FontSize', fontSize)
xlim([-1e-13, 1e-13])
ylim([-80, 80])
grid on

subplot(3,2,5)
plot(real(eigval_M_bf),imag(eigval_M_bf),'xm')
title('eigenvalues of $M_{BF}$', 'Interpreter', 'latex','FontSize', fontSize)
xlim([-150, 20])
ylim([-80, 80])
grid on

subplot(3,2,6)
plot(real(eigval_M_bf),imag(eigval_M_bf),'xm')
title('eigenvalues of $M_{BF}$', 'Interpreter', 'latex','FontSize', fontSize)
xlim([-1e-13, 1e-13])
ylim([-80, 80])
grid on


%   -----------------------
%-- Plots state
%   -----------------------

frame = length(tspan);

figure(4)

% phi

subplot(2,4,1)
plot(xspan,solve_state(1:Nx,frame),'+m')

subplot(2,4,2)
stem(frequences, abs(fft(solve_state(1:Nx,frame))),'+m')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,100])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,3)
stem(frequences, abs(fft(solve_state(1:Nx,frame))),'+m')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,4)
semilogy(frequences, abs(fft(solve_state(1:Nx,frame))),'+m')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-15 1e5])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

% eta

subplot(2,4,5)
plot(xspan,solve_state(Nx+1:2*Nx,frame),'+m')

subplot(2,4,6)
stem(frequences, abs(fft(solve_state(Nx+1:2*Nx,frame))),'+m')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,100])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,7)
stem(frequences, abs(fft(solve_state(Nx+1:2*Nx,frame))),'+m')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,8)
semilogy(frequences, abs(fft(solve_state(Nx+1:2*Nx,frame))),'+m')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-15,1e5])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$x$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

%   -----------------------
%-- Plots error
%   -----------------------

nb_frames = 100;
frame_list = 1:floor(length(tspan)/nb_frames):length(tspan);

i = length(frame_list);

frame = frame_list(i);

figure(5)

x0=1;
y0=1;
width=1600;
height=1200;
set(gcf,'position',[x0,y0,width,height])

% convergence

subplot(2,4,[1 5])
semilogy(tspan(frame_list(1:i)),norm_error(frame_list(1:i)),'-o',tspan(frame_list(1:i)),norm_error_bf(frame_list(1:i)),'-x',tspan(frame_list),error_th(frame_list),'-k')
xlabel('time $t$', 'Interpreter', 'latex','FontSize', fontSize)
ylim([1e-15 1e5])
grid on

% phi

subplot(2,4,2)
plot(frequences, abs(fft(solve_error(1:Nx,frame))),'o',frequences, abs(fft(solve_error_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,100])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$(\hat{x} - x)$ (RK4)','$(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,3)
plot(frequences, abs(fft(solve_error(1:Nx,frame))),'o',frequences, abs(fft(solve_error_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$(\hat{x} - x)$ (RK4)','$(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,4)
semilogy(frequences, abs(fft(solve_error(1:Nx,frame))),'o',frequences, abs(fft(solve_error_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-5 1e5])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$(\hat{x} - x)$ (RK4)','$(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

% eta

subplot(2,4,6)
plot(frequences, abs(fft(solve_error(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(solve_error_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,100])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$(\hat{x} - x)$ (RK4)','$(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,7)
plot(frequences, abs(fft(solve_error(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(solve_error_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$(\hat{x} - x)$ (RK4)','$(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,8)
semilogy(frequences, abs(fft(solve_error(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(solve_error_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-5,1e5])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$(\hat{x} - x)$ (RK4)','$(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

pause(0.01)


%%
if film
%   --------------
%-- Film 
%   --------------

for i = 1:nb_frames

frame = frame_list(i);

figure(6)

x0=1;
y0=1;
width=1600;
height=1200;
set(gcf,'position',[x0,y0,width,height])

% convergence

subplot(2,4,[1 5])
semilogy(tspan(frame_list(1:i)),norm_error(frame_list(1:i)),'-o',tspan(frame_list(1:i)),norm_error_bf(frame_list(1:i)),'-x',tspan(frame_list),error_th(frame_list),'-k')
xlabel('time $t$', 'Interpreter', 'latex','FontSize', fontSize)
ylim([1e-5 1e5])
grid on

% phi

subplot(2,4,2)
plot(frequences, abs(fft(solve_error(1:Nx,frame))),'o',frequences, abs(fft(solve_error_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,100])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$P_{BF}(\hat{x} - x)$ (RK4)','$P_{BF}(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,3)
plot(frequences, abs(fft(solve_error(1:Nx,frame))),'o',frequences, abs(fft(solve_error_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$P_{BF}(\hat{x} - x)$ (RK4)','$P_{BF}(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,4)
semilogy(frequences, abs(fft(solve_error(1:Nx,frame))),'o',frequences, abs(fft(solve_error_bf(1:Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-5 1e5])
title('frequencies on $\phi$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$P_{BF}(\hat{x} - x)$ (RK4)','$P_{BF}(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

% eta

subplot(2,4,6)
plot(frequences, abs(fft(solve_error(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(solve_error_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-2,100])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$P_{BF}(\hat{x} - x)$ (RK4)','$P_{BF}(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,7)
plot(frequences, abs(fft(solve_error(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(solve_error_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([-0.01,0.25])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$P_{BF}(\hat{x} - x)$ (RK4)','$P_{BF}(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

subplot(2,4,8)
semilogy(frequences, abs(fft(solve_error(Nx+1:2*Nx,frame))),'o',frequences, abs(fft(solve_error_bf(Nx+1:2*Nx,frame))),'x')
xlim([-floor(Nx/2)-1,floor(Nx/2)+1])
ylim([1e-5 1e5])
title('frequencies on $\eta$', 'Interpreter', 'latex','FontSize', fontSize)
legend('$P_{BF}(\hat{x} - x)$ (RK4)','$P_{BF}(\hat{x}_{BF} - x)$ (RK4)', 'Interpreter', 'latex','FontSize', fontSize)
grid on

pause(0.01)
    
end

end

%% epiphany graphs

Z0 = Lmat * C * E0;

figure(10)

subplot(2,4,1)
plot(xspan,E0(1:Nx),'b')
title("$\phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-2, 2])
grid on

subplot(2,4,2)
plot(xspan,Z0(1:Nx),'r')
title("$\gamma C^* C \phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-gain, gain])
grid on

subplot(2,4,3)
plot(xspan,E0(Nx+1:2*Nx),'b')
title("$\eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-2, 2])
grid on

subplot(2,4,4)
plot(xspan,Z0(Nx+1:2*Nx),'r')
title("$\gamma C^* C \eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-gain, gain])
grid on

subplot(2,4,5)
stem(frequences,abs(fft(E0(1:Nx))),'b')
title("$\phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
xlabel('frequencies')
grid on

subplot(2,4,6)
stem(frequences,abs(fft(Z0(1:Nx))),'r')
title("$\gamma C^* C \phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
xlabel('frequencies')
grid on

subplot(2,4,7)
stem(frequences,abs(fft(E0(Nx+1:2*Nx))),'b')
title("$\eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
xlabel('frequencies')
grid on

subplot(2,4,8)
stem(frequences,abs(fft(Z0(Nx+1:2*Nx))),'r')
title("$\gamma C^* C \eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
xlabel('frequencies')
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

% function Pn = ProjBF(Nx,dx,Nbf)
%     frequences = fftfreq(Nx,dx);
%     Low = diag(abs(frequences) <= Nbf);
%     e_kn_1 = exp(-i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
%     e_kn_2 = (1/Nx)*exp(i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
%     Pn = real(e_kn_2 * Low * e_kn_1);
% end

function Pn = ProjBF(Nx,dx,Nbf)
    frequences = fftfreq(Nx,dx);
    Low = diag(abs(frequences) <= Nbf);
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