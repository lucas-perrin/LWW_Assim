clear all
close all

fontSize = 16;

%% COMPUTATIONS

%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 128;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.8;       % value between 0 and 1
g        = 9.80665;   % gravity constant
T_start  = 0;         % start time
T        = 30;        % time window
%Nif_s    = 10;
Nif      = 5;         % Number of wave frequency in the solution wave
Nbf      = 5;

gain     = 1;

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

%   --------------------
%-- Initial condition
%   --------------------



%   -----------------------
%-- Solver
%   -----------------------

fprintf('...solving error equation BF RK4... \n')
[~,solve_error_zero] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,U(0));

gain_list = 10.^linspace(-2,4,50);
betas_    = zeros(1,length(gain_list));
t_f       = zeros(1,length(gain_list));

pp = [floor(length(tspan)/2),length(tspan)];

nb_points = floor(100 * T/50);
points_list = 1:floor(length(tspan)/nb_points):length(tspan);

if 1

f = waitbar(0,'Name','convergence rate vs gain','solving for each gain ...');

for i = 1:length(gain_list)
    M_bf       = A - PN * gain_list(i) * C' * C;
    %fprintf('...solving error equation BF RK4... \n')
    %fprintf('gain = %d \n',gain_list(i))
    [~,solve_error_zero] = odeRK4_inhom_ufunc(sparse(M_bf),G_zero,tspan,U(0));
    error_zero   = vecnorm(solve_error_zero);
    betas_(i)    = (log(error_zero(pp(2))) - log(error_zero(pp(1))))/(tspan(pp(2)) - tspan(pp(1)));
    t_f(i)       = error_zero(end);
    t_f(i)
    %fprintf('beta = %d \n\n',betas_(i))
    f = waitbar(i/length(gain_list),f,'solving for each gain ...');
end
close(f);

figure(4)
semilogx(gain_list,betas_,gain_list,t_f,'LineWidth',1.5)
title(['convergence rate $\beta$ vs gain $\gamma$, $|\omega| = $',num2str(size_obs),'$|\Omega|$'], 'Interpreter', 'latex','FontSize', fontSize)
xlabel('gain $\gamma$', 'Interpreter', 'latex','FontSize', fontSize)
ylabel('convergence rate $\beta$', 'Interpreter', 'latex','FontSize', fontSize)
grid on
saveas(gcf,['beta_vs_gain_',num2str(size_obs),'.png'])
end

%   -----------------------
%-- Convergence plot
%   -----------------------
pp = [floor(length(tspan)/2),length(tspan)];

error_zero = vecnorm(solve_error_zero);

points_reg = floor(rand(1,10).*length(tspan));

beta  = [ones(length(tspan),1) log(error_zero')]\tspan';

beta2 = [ones(length(tspan(points_reg)),1) error_zero(points_reg)']\tspan(points_reg)';

beta3 = [ones(length(tspan(pp)),1) log(error_zero(pp)')]\tspan(pp)';

gain
error_zero(end)

figure(3)
semilogy(tspan,error_zero,'-k',...
    tspan(points_reg),error_zero(points_reg),'o',...
    tspan(pp),error_zero(pp),'o',...
    tspan,exp(beta(2).*tspan),'-',...
    tspan,exp(beta2(2).*tspan),'-',...
    tspan,exp(beta3(2).*tspan),'-',...
    'LineWidth',1.5)
title(['size obs $=$ ',num2str(size_obs),', $N_{bf} = $',num2str(Nbf),', $N_{if} = N_{Fourier} =$ ',num2str(Nif)], 'Interpreter', 'latex','FontSize', fontSize)
ylim([1e-12 1e2])
legend('error','error points 2','error points 3','reg $\beta_1$','reg $\beta_2$','reg $\beta_3$', 'Interpreter', 'latex','FontSize', fontSize)
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