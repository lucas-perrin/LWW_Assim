clear all
close all

fontSize = 16;

%%


% Observer parameters

x0       = 0;           % start of space domain
Nx       = 128;         % Number of points in space domain
Lx       = 1;           % length of space domain
dt       = 1e-5;        % timestep
size_obs = 1;           % value between 0 and 1
g        = 9.81;        % gravity constant
T_start  = 15;          % start time
T        = 6;           % time window
T_end    = T_start + T; % end time

p        = 16;  % #CPU
gain     = 10;  % gamma parameter


% space discretization
xspan = linspace(x0, x0+Lx, Nx); % space grid
dx    = xspan(2) - xspan(1);    % step in space
dxsp  = Lx/(Nx-1);
xsp   = 0:dxsp:Lx;

A_dz = Matrix_A_dz(Nx,dxsp);

% solution
N_s     = 10;
a_s     = ones(1,N_s);
kw_s    = 2*pi*[1:N_s]/(Lx+dx);
omega_s = sqrt(kw_s.*g);

eta_s   = @(t) sum(a_s'.*sin(omega_s'.*t - kw_s'*xsp))';
phi0_s  = @(t) sum(a_s'.*(g.*diag(1./omega_s)*cos(omega_s'.*t - kw_s'.*xsp)))';

eta_s_dt  = @(t) sum(a_s'.*(omega_s'.*cos(omega_s'.*t - kw_s'*xsp)))';
phi0_s_dt = @(t) sum(-g.*a_s'.*(sin(omega_s'.*t - kw_s'*xsp)))';

sinus_s = @(t) sum(a_s'.*sin(omega_s'.*t - kw_s'*xsp))';

frequences = fftfreq(Nx,dx);

A1 = 1; A2 = 1; A3 = 1; A4 = 1;
k1 = 2*pi; k2 = 2*pi*2; k3 = 2*pi*3; k4 = 2*pi*4;
phi1 = 0; phi2 = pi; phi3 = 0; phi4 = pi;

% calculation of eta

eta_1 = A1*cos(k1*xspan - phi1);
eta_2 = A2*cos(k2*xspan - phi2);
eta_3 = A3*cos(k3*xspan - phi3);
eta_4 = A4*cos(k4*xspan - phi4);
eta = (A1*cos(k1*xspan - phi1) + A2*cos(k2*xspan - phi2) + A3*cos(k3*xspan - phi3) + A4*cos(k4*xspan - phi4))';

Nbf = 10;

Pn = ProjBF(Nx,dx,Nbf);

frequences = fftfreq(Nx,dx);
Low = diag(abs(frequences) < Nbf);

figure(1)
subplot(2,1,1)
plot(xspan,eta,xspan,Pn*eta)
subplot(2,1,2)
plot(xspan,eta_1,xspan,eta_2,xspan,eta_3,xspan,eta_4)
legend('eta 1','eta 2','eta 3','eta 4')

figure(2)
plot(fftfreq(Nx,dx),fft(eta),'x',fftfreq(Nx,dx),Low*fft(eta),'o')

fprintf('|eta - Pn*eta| = %d \n',norm(Pn*eta - eta)/norm(eta))


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