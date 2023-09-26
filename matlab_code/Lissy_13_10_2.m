clear all
close all

fontSize = 16;

%% COMPUTATIONS

%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 2^7;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.5;       % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 0;         % start time
T        = 150;        % time window
%Nif_s    = 10;
Nif      = 2;         % Number of wave frequency in the solution wave
Nbf      = 128;

%gain     = floor(exp(2*Nbf+1));     % gamma parameter
gain     = 0.5;
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
omega = sqrt(kw.*g);    % time shift (to be accurate on a physics level)

% solution
eta = @(t) sum(a'.*sin(omega'.*t - kw'*xspan))';
phi = @(t) sum(a'.*(g.*diag(1./omega)*cos(omega'.*t - kw'.*xspan)))';
U   = @(t) [phi(t); eta(t)];

% derivative of the solution
eta_dt = @(t) sum(a'.*(omega'.*cos(omega'.*t - kw'*xspan)))';
phi_dt = @(t) sum(-g.*a'.*(sin(omega'.*t - kw'*xspan)))';
U_dt     = @(t) [phi_dt(t); eta_dt(t)];

% % %   -----------------------
% % %-- True solution grid
% % %   -----------------------
% % U_mat = zeros(2*Nx,Nt);
% % for t=1:Nt
% %     U_mat(:,t) = U(tspan(t));
% % end

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
%Lmat       = C';
M          = A - Lmat * C;

% low frequencies
M_bf       = A - PN * Lmat * C;

%   -----------------------
%-- Observer source term
%   -----------------------
Gobs     = @(t) Lmat * C * U(t);

%low frequencies
Gobs_bf = @(t) PN * Lmat * C * U(t);


%   -----------------------
%-- Zero source term
%   -----------------------
G_zero = @(t) zeros(2*Nx,1);

figure(1)
subplot(1,2,1)
plot(xspan,phi(0),...
    xspan,ones(1,Nx).*mean(phi(0)),'--c','LineWidth',1.5)
xlabel('$x$', 'Interpreter', 'latex','FontSize', fontSize)
title('$\phi(0)$', 'Interpreter', 'latex','FontSize', fontSize)
subplot(1,2,2)
plot(xspan,eta(0),...
    xspan,ones(1,Nx).*mean(eta(0)),'--c','LineWidth',1.5)
xlabel('$x$', 'Interpreter', 'latex','FontSize', fontSize)
title('$\eta(0)$', 'Interpreter', 'latex','FontSize', fontSize)

%   --------------------
%-- Initial condition
%   --------------------

f1 = zeros(1,Nx);
f2 = zeros(1,Nx);

for i = 1:Nif
    f1 = f1 + sin((i)*2*pi*xspan/L);
    f2 = f2 + sin((i)*2*pi*xspan/L);
end

F1 = zeros(1,Nx);
F2 = zeros(1,Nx);

% for k = 1:Nif
%     F1 = F1 + sin((Nx/2 + 1 - k)*2*pi*xspan/L);
%     F2 = F2 + sin((Nx/2 + 1 - k)*2*pi*xspan/L);
% end

F1 = sin((Nx/2)*2*pi*xspan/L) - sin((Nx/2 - 1)*2*pi*xspan/L);
F2 = F1;

figure(2)
subplot(1,2,1)
plot(xspan,f1,'b',xspan,F1,'r',...
    xspan,ones(1,Nx).*mean(f1),'--m',...
    xspan,ones(1,Nx).*mean(F1),'--c','LineWidth',1.5)
xlabel('$x$', 'Interpreter', 'latex','FontSize', fontSize)
title('$\phi(0)$', 'Interpreter', 'latex','FontSize', fontSize)
subplot(1,2,2)
plot(xspan,f2,'b',xspan,F2,'r',...
    xspan,ones(1,Nx).*mean(f2),'--c',...
    xspan,ones(1,Nx).*mean(F2),'--m','LineWidth',1.5)
xlabel('$x$', 'Interpreter', 'latex','FontSize', fontSize)
title('$\eta(0)$', 'Interpreter', 'latex','FontSize', fontSize)

%   -----------------------
%-- Solver
%   -----------------------

e0 = [f1' ; f2'];
E0 = [F1' ; F2'];

fprintf('...solving error equation BF RK4... \n')
[~,solve_error_low] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,e0);

fprintf('...solving error equation BF RK4... \n')
[~,solve_error_hig] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,E0);

nb_points = floor(100 * T/50);
points_list = 1:floor(length(tspan)/nb_points):length(tspan);

%   -----------------------
%-- Convergence plot
%   -----------------------
error_low = vecnorm(solve_error_low);

error_hig = vecnorm(solve_error_hig);

figure(3)
semilogy(tspan(points_list),error_low(points_list),'-b',...
    tspan(points_list),error_hig(points_list),'-r','LineWidth',1.5)
title(['size obs $=$ ',num2str(size_obs),', $N_{bf} = $ ',num2str(Nbf),', $N_{if} = N_{Fourier} =$ ',num2str(Nif)], 'Interpreter', 'latex','FontSize', fontSize)
ylim([1e-12 1e2])
legend('$\varepsilon(t)$ low','$\varepsilon(t)$ high', 'Interpreter', 'latex','FontSize', fontSize)
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

function F=Fourier(x,f,N,L)

T = L/(max(x) - min(x));

n   = 0;
a_0 = 1/T*trapz(x,f.*cos(n.*x.*(2*pi)./T));

for n = 1:N
    a(n) = 2/T*trapz(x,f.*cos(n.*x.*(2*pi)./T));
    b(n) = 2/T*trapz(x,f.*sin(n.*x.*(2*pi)./T));
end

F = a_0;

for k=1:N
    F = F + a(k).*cos(k.*x.*(2*pi)./T) + b(k).*sin(k.*x.*(2*pi)./T);
end

end