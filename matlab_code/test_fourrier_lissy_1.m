clear all
close all

fontSize = 16;

%%


% Observer parameters

x0       = 0;           % start of space domain
Nx       = 128;         % Number of points in space domain
L        = 1;           % length of space domain
dt       = 1e-5;        % timestep
size_obs = 1;           % value between 0 and 1
g        = 9.81;        % gravity constant
T_start  = 15;          % start time
T        = 6;           % time window
T_end    = T_start + T; % end time
N_s      = 10;          % Number of wave frequency in the solution wave

p        = 16;  % #CPU
gain     = 10;  % gamma parameter


% space discretization
xspan = linspace(x0, x0+L, Nx); % space grid
dx    = xspan(2) -xspan(1);      % step in space
dxsp  = L/(Nx-1);
xsp   = 0:dxsp:L;

A_dz = Matrix_A_dz(Nx,dxsp);

% solution
N_s     = 80;
a_s     = ones(1,N_s);
kw_s    = 2*pi*[1:N_s]/L;
omega_s = sqrt(kw_s.*g);

eta_s   = @(t) sum(a_s'.*sin(omega_s'.*t - kw_s'*xsp))';
phi0_s  = @(t) sum(a_s'.*(g.*diag(1./omega_s)*cos(omega_s'.*t - kw_s'.*xsp)))';

eta_s_dt  = @(t) sum(a_s'.*(omega_s'.*cos(omega_s'.*t - kw_s'*xsp)))';
phi0_s_dt = @(t) sum(-g.*a_s'.*(sin(omega_s'.*t - kw_s'*xsp)))';

tt = 35;

% plot
figure(1)
subplot(3,1,1)
plot(xspan,eta_s(tt))
title('$\eta(0)$','Interpreter', 'latex','FontSize',fontSize)
subplot(3,1,2)
plot(xspan,phi0_s(tt))
title('$\phi(0)$','Interpreter', 'latex','FontSize',fontSize)
subplot(3,1,3)
plot(xspan,ifft(fft(eta_s(tt))))
title('ifft$($fft$(\eta(0)))$','Interpreter', 'latex','FontSize',fontSize)

figure(2)
subplot(3,1,1)
plot(fftfreq(Nx,dx),real(fft(eta_s(tt))),'x')
subplot(3,1,2)
plot(fftfreq(Nx,dx),imag(fft(eta_s(tt))),'x')
subplot(3,1,3)
plot(fftfreq(Nx,dx),abs(fft(eta_s(tt))),'x')

% Projection

frequences = fftfreq(Nx,dx);
Proj_bf = zeros(Nx);
N_max = 10;
low_frequences = abs(frequences) < N_max;
eta_stt_fourrier_bf = diag(low_frequences)*fft(eta_s(tt));
eta_stt_bf = ifft(eta_stt_fourrier_bf);

figure(3)
subplot(3,1,1)
plot(xspan,eta_s(tt))
subplot(3,1,2)
plot(xspan,phi0_s(tt))
subplot(3,1,3)
plot(xspan,ifft(diag(low_frequences)*fft(eta_s(tt))),xspan,eta_s(tt))

figure(4)
subplot(2,1,1)
plot(fftfreq(Nx,dx),abs(fft(eta_s(tt))),'x')
subplot(2,1,2)
plot(fftfreq(Nx,dx),abs(eta_stt_fourrier_bf),'x')

% DFT matrcielle

e_kn_1 = exp(-i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
e_kn_2 = (1/Nx)*exp(i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
dtf_eta_s = e_kn_1*eta_s(tt);

figure(5)
subplot(2,1,1)
plot(fftfreq(Nx,dx),abs(fft(eta_s(tt))),'o',fftfreq(Nx,dx),abs(dtf_eta_s),'x')
subplot(2,1,2)
plot(xspan,eta_s(tt),'-',xspan,e_kn_2*dtf_eta_s,'--')

% Projection + DFT matricielle

PN = e_kn_2*diag(low_frequences)*e_kn_1;
PN = real(PN);

disp(norm(PN*eta_s(tt) - ifft(diag(low_frequences)*fft(eta_s(tt)))))

Nbf = 10;

figure(6)
subplot(3,1,1)
plot(fftfreq(Nx,dx),real(ProjBF(Nx,dx,Nbf)*eta_s(tt) - ifft(diag(low_frequences)*fft(eta_s(tt)))),'x')
subplot(3,1,2)
plot(fftfreq(Nx,dx),imag(ProjBF(Nx,dx,Nbf)*eta_s(tt) - ifft(diag(low_frequences)*fft(eta_s(tt)))),'x')
subplot(3,1,3)
plot(fftfreq(Nx,dx),abs(ProjBF(Nx,dx,Nbf)*eta_s(tt) - ifft(diag(low_frequences)*fft(eta_s(tt)))),'x')

figure(7)
subplot(3,1,1)
plot(fftfreq(Nx,dx),real(ProjBF(Nx,dx,Nbf)*eta_s(tt) - eta_s(tt)),'x')
subplot(3,1,2)
plot(fftfreq(Nx,dx),imag(ProjBF(Nx,dx,Nbf)*eta_s(tt) - eta_s(tt)),'x')
subplot(3,1,3)
plot(fftfreq(Nx,dx),abs(ProjBF(Nx,dx,Nbf)*eta_s(tt) - eta_s(tt)),'x')

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