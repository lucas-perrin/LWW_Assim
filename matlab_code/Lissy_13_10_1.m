clear all
close all

fontSize = 16;

%% COMPUTATIONS

%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 2^9;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.5;       % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 0;         % start time
T        = 20;        % time window
Nif      = 1;         % Number of wave frequency in the solution wave
Nbf      = Nx;

%gain     = floor(exp(2*Nbf+1));     % gamma parameter
gain     = 1;
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
omega = sqrt(kw.*g);  % time shift (to be accurate on a physics level)

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
%Lmat       = gain * C';
Lmat       = C';
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

f1 = 1./(2+sin(2*pi.*xspan./L));
f2 = 1./(2+sin(2*pi.*xspan./L));

f1 = f1 - mean(f1);
f2 = f2 - mean(f2);

F1 = Fourier(xspan,f1,Nif,L);
F2 = Fourier(xspan,f2,Nif,L);

figure(2)
subplot(1,2,1)
plot(xspan,f1,xspan,F1,...
    xspan,ones(1,Nx).*mean(f1),'--m',...
    xspan,ones(1,Nx).*mean(F1),'--c','LineWidth',1.5)
xlabel('$x$', 'Interpreter', 'latex','FontSize', fontSize)
title('$\phi(0)$', 'Interpreter', 'latex','FontSize', fontSize)
subplot(1,2,2)
plot(xspan,f2,xspan,F2,...
    xspan,ones(1,Nx).*mean(f2),'--c',...
    xspan,ones(1,Nx).*mean(F2),'--m','LineWidth',1.5)
xlabel('$x$', 'Interpreter', 'latex','FontSize', fontSize)
title('$\eta(0)$', 'Interpreter', 'latex','FontSize', fontSize)

%   -----------------------
%-- Solver
%   -----------------------

e0 = [f1' ; f2'];
E0 = [F1' ; F2'];

% fprintf('...solving error equation BF RK4... \n')
% [~,solve_error_c] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,e0);
% 
% fprintf('...solving error equation BF RK4... \n')
% [~,solve_error_f] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,E0);

e0 = [ones(Nx,1);ones(Nx,1)];

fprintf('...solving error equation BF RK4... \n')
[~,solve_error_c] = odeRK4_inhom_ufunc(sparse(M),G_zero,tspan,e0);

fprintf('...solving error equation BF RK4... \n')
[~,solve_error_f] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,e0);

nb_points = floor(100 * T/50);
points_list = 1:floor(length(tspan)/nb_points):length(tspan);

%   -----------------------
%-- Convergence plot
%   -----------------------
error_c = vecnorm(solve_error_c);
error_c = mean(solve_error_c,1);
error_f = vecnorm(solve_error_f);
error_f = mean(solve_error_f,1);

error_eta = mean(solve_error_c(1:Nx,:),1);
error_phi = mean(solve_error_c(Nx+1:2*Nx,:),1);

figure(3)
semilogy(tspan(points_list),abs(error_c(points_list)),'-b',...
    tspan(points_list),abs(error_f(points_list)),'--r','LineWidth',1.5)
title(['size obs $=$ ',num2str(size_obs),', $N_{bf} = $ ',num2str(Nbf),', $N_{if} = N_{Fourier} =$ ',num2str(Nif)], 'Interpreter', 'latex','FontSize', fontSize)
legend('$\varepsilon(t)$ for $f(x) = 2x-1$','$\varepsilon(t)$ for $S(f)_{N_{if}}(x)$', 'Interpreter', 'latex','FontSize', fontSize)
grid on

figure(4)
semilogy(tspan(points_list),abs(error_eta(points_list)),'-b',...
    tspan(points_list),abs(error_phi(points_list)),'-r','LineWidth',1.5)
title(['size obs $=$ ',num2str(size_obs),', $N_{bf} = $ ',num2str(Nbf),', $N_{if} = N_{Fourier} =$ ',num2str(Nif)], 'Interpreter', 'latex','FontSize', fontSize)
legend('mean phi','mean eta', 'Interpreter', 'latex','FontSize', fontSize)
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