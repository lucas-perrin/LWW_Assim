fontSize = 16;

%% COMPUTATIONS
compute = 1;

if compute
%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;         % start of space domain
Lx       = 1;         % length of spatial grid
Nx       = 2^4+1;       % number of points in space domain
dt       = 1e-3;      % timestep
size_obs = 0.5;       % value between 0 and 1
g        = 9.80665;   % gravity constant
g        = 1;
T_start  = 0;         % start time
T        = 250;       % time window

N_ini_fq = 4; % Number of wave frequency in the initial condition
%N_ini_fq = 0;
N_low_fq = 4; % Number of Low frenqcies filtered
Mean_fq  = 1;  % Do we filter out the mode 0 ? 1 Yes, 0 No
mean_ini = [0,0];  % add a mean to the initial condition ?

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
A_dz_2 = Matrix_A_dz_2(Nx,dx); % kernel matrix

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

%e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
if Mean_fq
    pn_low = Proj_low_wout_zero(Nx,dx,N_low_fq);
    pn_low_wout_zero = pn_low;

else
    pn_low = Proj_low(Nx,dx,N_low_fq);
    pn_low_wout_zero = Proj_low_wout_zero(Nx,dx,N_low_fq);
end
pn_high = Proj_high(Nx,dx,N_low_fq);
pn_zero = Proj_zero(Nx,dx);
pn_wout_zero = Proj_wout_zero(Nx,dx);
frequences = fftfreq(Nx,dx);

PN_low = [pn_low, zeros(Nx); zeros(Nx), pn_low];
PN_high = [pn_high, zeros(Nx); zeros(Nx), pn_high];
PN_wout_zero = [pn_wout_zero, zeros(Nx); zeros(Nx), pn_wout_zero];


%   -----------------------
%-- Observer setting
%   -----------------------
obs_map    = abs(xspan) <= size_obs;
obs_map    = abs(xspan - Lx/2) <= size_obs*Lx/2;
%obs_map    = ones(1,Nx);
[m_obs,Cc] = GetC(obs_map);
C          = [zeros(m_obs,Nx) Cc];

M          = A - gain * PN_wout_zero * (C') * C;
%M          = A - (C') * C;
%M          = [zeros(Nx,Nx), -eye(Nx); Matrix_A_dz(Nx,dx), - Cc' * Cc];
M_low      = A - gain * PN_low * (C') * C;
M_high     = A - gain * PN_high * (C') * C;

eig_M = eig(M);
cfactor = min(abs(eig_M(abs(imag(eig_M)) > 1e-10)));
place_cfactor = find(abs(eig_M) == min(abs(eig_M(abs(imag(eig_M)) > 1e-10))));
eig_cfactor = eig_M(place_cfactor);

convergence   = exp(-tspan.*abs(eig_cfactor(1)));

%   -----------------------
%-- Initial Condition
%   -----------------------

f1 = zeros(1,Nx);
f2 = zeros(1,Nx);

if N_ini_fq > 0
for i = 1:N_ini_fq
    f1 = f1 + sin((i)*2*pi*xspan/L);
    f2 = f2 + sin((i)*2*pi*xspan/L);
end
end
e0 = [f1 + mean_ini(1),f2 + mean_ini(2)];

%   -----------------------
%-- Solver
%   -----------------------
G_zero = @(t) zeros(2*Nx,1);

fprintf('...solving error equation... \n')
[~,solve_error] = odeRK4_inhom_ufunc(sparse(M),G_zero,tspan,e0);
fprintf('-> done \n')

fprintf('...solving error equation low... \n')
[~,solve_error_low] = odeRK4_inhom_ufunc(sparse(M_low),G_zero,tspan,e0);
fprintf('-> done \n')

% error
phi_error = solve_error(1:Nx,:);
eta_error = solve_error(Nx+1:2*Nx,:);

phi_error_low = solve_error_low(1:Nx,:);
eta_error_low = solve_error_low(Nx+1:2*Nx,:);

% mode 0 error
mean_phi_error = mean(phi_error,1);
mean_eta_error = mean(eta_error,1);

mean_phi_error_low = mean(phi_error_low,1);
mean_eta_error_low = mean(eta_error_low,1);

% vecnorm error
vec_phi_error = vecnorm(phi_error);
vec_eta_error = vecnorm(eta_error);

vec_phi_error_low = vecnorm(phi_error_low);
vec_eta_error_low = vecnorm(eta_error_low);

% vecnorm zero freq
zero_vec_phi_error = vecnorm(pn_zero * phi_error);
zero_vec_eta_error = vecnorm(pn_zero * eta_error);

zero_vec_phi_error_low = vecnorm(pn_zero * phi_error_low);
zero_vec_eta_error_low = vecnorm(pn_zero * eta_error_low);

% vecnorm low freq
low_vec_phi_error = vecnorm(pn_low_wout_zero * phi_error);
low_vec_eta_error = vecnorm(pn_low_wout_zero * eta_error);

low_vec_phi_error_low = vecnorm(pn_low_wout_zero * phi_error_low);
low_vec_eta_error_low = vecnorm(pn_low_wout_zero * eta_error_low);

% vecnorm high freq
high_vec_phi_error = vecnorm(pn_high * phi_error);
high_vec_eta_error = vecnorm(pn_high * eta_error);

high_vec_phi_error_low = vecnorm(pn_high * phi_error_low);
high_vec_eta_error_low = vecnorm(pn_high * eta_error_low);

% vecnorm all freq without mode 0
out_zero_vec_phi_error = vecnorm(pn_wout_zero * phi_error);
out_zero_vec_eta_error = vecnorm(pn_wout_zero * eta_error);

out_zero_vec_phi_error_low = vecnorm(pn_wout_zero * phi_error_low);
out_zero_vec_eta_error_low = vecnorm(pn_wout_zero * eta_error_low);

z = polyfit(tspan,log(abs(vec_eta_error)),1);
conv = z(1);

convergence2   = exp(tspan.*conv);

end

%% Plots
nb_points = floor(100 * T/50);
nb_points = 100;
points_list = 1:floor(length(tspan)/nb_points):length(tspan);

figure(1)
semilogy(tspan(points_list),abs(mean_phi_error(points_list)),'--ro',...
    tspan(points_list),abs(mean_eta_error(points_list)),'-rd',...
    tspan(points_list),abs(mean_phi_error_low(points_list)),'--bx',...
    tspan(points_list),abs(mean_eta_error_low(points_list)),'-bs',...
    'LineWidth',1.5)
title(['MEAN OF ERROR : size obs $=$ ',num2str(size_obs),', $N_{low} = $ ',num2str(N_low_fq),', $N_{ini} =$ ',num2str(N_ini_fq)], 'Interpreter', 'latex','FontSize', fontSize)
legend('mean phi','mean eta', 'Interpreter', 'latex','FontSize', fontSize)
grid on

figure(2)
semilogy(tspan(points_list),abs(vec_phi_error(points_list)),'--ro',...
    tspan(points_list),abs(vec_eta_error(points_list)),'-rd',...
    tspan(points_list),abs(vec_phi_error_low(points_list)),'--bx',...
    tspan(points_list),abs(vec_eta_error_low(points_list)),'-bs',...
    tspan(points_list),convergence(points_list),'-',...
    'LineWidth',1.5)
title(['NORM ERROR ON ALL FREQS : size obs $=$ ',num2str(size_obs),', $N_{low} = $ ',num2str(N_low_fq),', $N_{ini} =$ ',num2str(N_ini_fq)], 'Interpreter', 'latex','FontSize', fontSize)
legend('mean phi','mean eta', 'Interpreter', 'latex','FontSize', fontSize)
grid on

figure(3)
semilogy(tspan(points_list),abs(low_vec_phi_error(points_list)),'--ro',...
    tspan(points_list),abs(low_vec_eta_error(points_list)),'-rd',...
    tspan(points_list),abs(low_vec_phi_error_low(points_list)),'--bx',...
    tspan(points_list),abs(low_vec_eta_error_low(points_list)),'-bs',...
    'LineWidth',1.5)
title(['NORM ERROR ON LOW FREQS : size obs $=$ ',num2str(size_obs),', $N_{low} = $ ',num2str(N_low_fq),', $N_{ini} =$ ',num2str(N_ini_fq)], 'Interpreter', 'latex','FontSize', fontSize)
legend('mean phi','mean eta', 'Interpreter', 'latex','FontSize', fontSize)
grid on

figure(4)
semilogy(tspan(points_list),abs(high_vec_phi_error(points_list)),'--ro',...
    tspan(points_list),abs(high_vec_eta_error(points_list)),'-rd',...
    tspan(points_list),abs(high_vec_phi_error_low(points_list)),'--bx',...
    tspan(points_list),abs(high_vec_eta_error_low(points_list)),'-bs',...
    'LineWidth',1.5)
title(['NORM ERROR ON HIGH FREQS : size obs $=$ ',num2str(size_obs),', $N_{low} = $ ',num2str(N_low_fq),', $N_{ini} =$ ',num2str(N_ini_fq)], 'Interpreter', 'latex','FontSize', fontSize)
legend('mean phi','mean eta', 'Interpreter', 'latex','FontSize', fontSize)
grid on

figure(5)
semilogy(tspan(points_list),abs(zero_vec_phi_error(points_list)),'--ro',...
    tspan(points_list),abs(zero_vec_eta_error(points_list)),'-rd',...
    tspan(points_list),abs(zero_vec_phi_error_low(points_list)),'--bx',...
    tspan(points_list),abs(zero_vec_eta_error_low(points_list)),'-bs',...
    'LineWidth',1.5)
title(['NORM ERROR ON ZERO FREQ : size obs $=$ ',num2str(size_obs),', $N_{low} = $ ',num2str(N_low_fq),', $N_{ini} =$ ',num2str(N_ini_fq)], 'Interpreter', 'latex','FontSize', fontSize)
legend('mean phi','mean eta', 'Interpreter', 'latex','FontSize', fontSize)
grid on

figure(6)
semilogy(tspan(points_list),abs(out_zero_vec_phi_error(points_list)),'--ro',...
    tspan(points_list),abs(out_zero_vec_eta_error(points_list)),'-rd',...
    tspan(points_list),abs(out_zero_vec_phi_error_low(points_list)),'--bx',...
    tspan(points_list),abs(out_zero_vec_eta_error_low(points_list)),'-bs',...
    'LineWidth',1.5)
title(['NORM ALL FREQ WITOUHT ZERO : size obs $=$ ',num2str(size_obs),', $N_{low} = $ ',num2str(N_low_fq),', $N_{ini} =$ ',num2str(N_ini_fq)], 'Interpreter', 'latex','FontSize', fontSize)
legend('mean phi','mean eta', 'Interpreter', 'latex','FontSize', fontSize)
grid on

fprintf('\n')

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
    for j = 0:Nx-1
        A_dz(j+1,:) = (1/Nx)*circshift(kernel,j);
    end
end

function A_dz_2 = Matrix_A_dz_2(Nx,dx)
    e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    frequences = fftfreq(Nx,dx);
    A_dz_2 = real(e_kn_2 * diag(abs(frequences)) * e_kn_1) * 2 * pi;
end

function Pn = Proj_low(Nx,dx,Nbf)
    frequences = fftfreq(Nx,dx);
    Low = diag(abs(frequences) < Nbf);
    e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    Pn = real(e_kn_2 * Low * e_kn_1);
end

function Pn = Proj_high(Nx,dx,Nbf)
    frequences = fftfreq(Nx,dx);
    Low = diag(abs(frequences) >= Nbf);
    e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    Pn = real(e_kn_2 * Low * e_kn_1);
end

function Pn = Proj_low_wout_zero(Nx,dx,Nbf)
    frequences = fftfreq(Nx,dx);
    Low = diag((abs(frequences) > 0).*(abs(frequences) < Nbf));
    e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    Pn = real(e_kn_2 * Low * e_kn_1);
end

function Pn = Proj_zero(Nx,dx)
    frequences = fftfreq(Nx,dx);
    Low = diag((abs(frequences) == 0));
    e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    Pn = real(e_kn_2 * Low * e_kn_1);
end

function Pn = Proj_wout_zero(Nx,dx)
    frequences = fftfreq(Nx,dx);
    Low = diag((abs(frequences) > 0));
    e_kn_1 = exp(-1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
    e_kn_2 = (1/Nx)*exp(1i*2*pi*[0:Nx-1]'*[0:Nx-1]/Nx);
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