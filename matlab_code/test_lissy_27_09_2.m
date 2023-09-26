clear all
close all

fontSize = 16;
film = 0;

%%

% parameters to tune :
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

gain     = floor(exp(2*Nbf+1));     % gamma parameter
p        = 16;     % #CPU

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
e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
Pn_1 = ProjBF(Nx,dx,Nbf);
Pn_2 = ProjBF_2(Nx,dx,Nbf);
frequences = fftfreq(Nx,dx);
PN_1 = [Pn_1, zeros(Nx); zeros(Nx), Pn_1];
PN_2 = [Pn_2, zeros(Nx); zeros(Nx), Pn_2];

%   --------------------
%-- Initial condition
%   --------------------

e0_phi = ones(1,Nx);
e0_eta = ones(1,Nx);

% for i = 1:Nif
%     e0_phi = e0_phi + sin(i*2*pi*xspan/L);
%     e0_eta = e0_eta + sin(i*2*pi*xspan/L);
% end

E0 = [e0_phi' ; e0_eta'];

%   --------------------
%-- Zero source term
%   --------------------

G_zero = @(t) zeros(2*Nx,1);

%   --------------------
%-- Observer setting
%   --------------------

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

M_bf       = A - PN_1 * Lmat * C;

%   --------------------
%-- Plots
%   --------------------

Y0  = Lmat * C * E0;
Y0f = [abs(fft(Y0(1:Nx))); abs(fft(Y0(Nx+1:2*Nx)))];
Z0_1  = PN_1 * Lmat * C * E0;
Z0_1f = [abs(fft(Z0_1(1:Nx))); abs(fft(Z0_1(Nx+1:2*Nx)))];
ZZ0_1 = Z0_1 - [ones(Nx,1).*mean(Z0_1(1:Nx)); ones(Nx,1).*mean(Z0_1(Nx+1:2*Nx))];
ZZ0_1f = [abs(fft(ZZ0_1(1:Nx))); abs(fft(ZZ0_1(Nx+1:2*Nx)))];
Z0_2  = PN_2 * Lmat * C * E0;
Z0_2f = [abs(fft(Z0_2(1:Nx))); abs(fft(Z0_2(Nx+1:2*Nx)))];

%%

figure(1)

subplot(7,4,1)
plot(xspan,E0(1:Nx),'b')
title("$\phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])

subplot(7,4,2)
plot(xspan,E0(Nx+1:2*Nx),'b')
title("$\eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])

subplot(7,4,3)
stem(frequences,abs(fft(Pn_2*E0(1:Nx))),'b')
title("DFT($P_{BF} \phi_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,4)
stem(frequences,abs(fft(Pn_2*E0(Nx+1:2*Nx))),'b')
title("DFT($P_{BF} \eta_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,5)
plot(xspan,Y0(1:Nx),'r')
title("$C^* C \phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])


subplot(7,4,6)
plot(xspan,Y0(Nx+1:2*Nx),'r')
title("$C^* C \eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])


subplot(7,4,9)
stem(frequences,Y0f(1:Nx),'r')
title("DFT($C^* C \phi_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,10)
stem(frequences,Y0f(Nx+1:2*Nx),'r')
title("DFT($C^* C \eta_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,13)
stem(frequences,Z0_1f(1:Nx),'m')
title("DFT($P_{BF} C^* C \phi_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,14)
stem(frequences,Z0_1f(Nx+1:2*Nx),'m')
title("DFT($P_{BF} C^* C \eta_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,17)
plot(xspan,Z0_1(1:Nx),'m')
title("$P_{BF} C^* C \phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])


subplot(7,4,18)
plot(xspan,Z0_1(Nx+1:2*Nx),'m')
title("$P_{BF} C^* C \eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])


subplot(7,4,21)
stem(frequences,ZZ0_1f(1:Nx),'k')
title("DFT($(P_{BF} C^* C \phi_0) - $mean($P_{BF} C^* C \phi_0$))", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,22)
stem(frequences,ZZ0_1f(Nx+1:2*Nx),'k')
title("DFT($(P_{BF} C^* C \eta_0) - $mean($P_{BF} C^* C \eta_0$))", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,25)
plot(xspan,ZZ0_1(1:Nx),'k')
title("$(P_{BF} C^* C \phi_0) - $mean($P_{BF} C^* C \phi_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])


subplot(7,4,26)
plot(xspan,ZZ0_1(Nx+1:2*Nx),'k')
title("$(P_{BF} C^* C \eta_0) - $mean($P_{BF} C^* C \eta_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])


subplot(7,4,15)
stem(frequences,Z0_2f(1:Nx),'k')
title("DFT($P_{BF2} C^* C \phi_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,16)
stem(frequences,Z0_2f(Nx+1:2*Nx),'k')
title("DFT($P_{BF2} C^* C \eta_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(7,4,19)
plot(xspan,Z0_2(1:Nx),'k')
title("$P_{BF2} C^* C \phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])


subplot(7,4,20)
plot(xspan,Z0_2(Nx+1:2*Nx),'k')
title("$P_{BF2} C^* C \eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])

%%

Y0phi = Y0(1:Nx);
Y0eta = Y0(Nx+1:2*Nx);

figure(2)

subplot(3,4,1)
plot(xspan,E0(1:Nx),'b')
title("$\phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])

subplot(3,4,2)
plot(xspan,E0(Nx+1:2*Nx),'b')
title("$\eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])

subplot(3,4,3)
stem(frequences,abs(fft(E0(1:Nx))),'b')
title("DFT($\phi_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(3,4,4)
stem(frequences,abs(fft(E0(Nx+1:2*Nx))),'b')
title("DFT($\eta_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(3,4,5)
plot(xspan,Y0phi,'r')
title("$C^* C \phi_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])

subplot(3,4,6)
plot(xspan,Y0eta,'r')
title("$C^* C \eta_0$", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-Nif, Nif])
xlim([0, 1])

subplot(3,4,7)
stem(frequences,Y0f(1:Nx),'r')
title("DFT($C^* C \phi_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(3,4,8)
stem(frequences,Y0f(Nx+1:2*Nx),'r')
title("DFT($C^* C \eta_0$)", 'Interpreter', 'latex','FontSize', fontSize)
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(3,4,9)
%plot(xspan(obs_map),Y0phi(obs_map),'r')
ylim([-Nif, Nif])
xlim([0, 1])

subplot(3,4,10)
plot(xspan(obs_map),Y0eta(obs_map),'r')
ylim([-Nif, Nif])
xlim([0, 1])

subplot(3,4,11)
stem(fftfreq(m_obs,dx),abs(fft(Y0phi(obs_map))),'r')
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
grid on

subplot(3,4,12)
stem(fftfreq(m_obs,dx),abs(fft(Y0eta(obs_map))),'r')
ylim([-1, 100])
xlim([-Nx/2 - 1,Nx/2+1])
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