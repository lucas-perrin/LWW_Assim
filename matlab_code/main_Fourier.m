clear all
close all

% ------------------------------
% graphics settings
% ------------------------------

Fontsize = 18;
Fontsize_label = 22;
Fontsize_axes = 18;
Linesize = 2;
Marksize = 9;

% ------------------------------
% Parameters to tune
% ------------------------------

x0 = 0;
L = 2 * pi;
Nx = 2^5 + 1;
Nf = floor(Nx / 2);
dt = 1e-3;
a = 3/2;
g = 9.80665;
gain = 1;
T_start = 0;
T = 250;
N_ini_fq = 2^2;

% ------------------------------
% Space discretization
% ------------------------------

dx = L / Nx;
Lx = L - dx;
xspan = x0:dx:Lx;
frequencies = fftfreq(Nx, dx)*2*pi;

% ------------------------------
% Time discretization
% ------------------------------

T_end = T_start + T;
tspan = T_start:dt:T_end;
Nt = length(tspan);

% ------------------------------
% Matrix computations
% ------------------------------

% 1) DFT and iDFT Matrices (as well as no mean matrices)
DFT = 1 / Nx * getDFT(Nx);
iDFT = getiDFT(Nx);

% 2) True matrix M not in Fourier
O = zeros(Nx, Nx);
I = eye(Nx);
G = Matrix_A_dz(Nx, dx);
[~, C] = GetC(abs(xspan - pi) <= a);
[~, C] = GetC(abs(xspan) <= 2*a);
obs_space = abs(xspan - pi) <= a;
LC = - (C' * C);
M = sparse([O, -g * I; G, gain * LC]);

zz = iDFT*LC*DFT;

% True matrix M in Fourier
fO = zeros(2 * Nf + 1, 2 * Nf + 1);
fI = eye(2 * Nf + 1);
fG = diag(abs(frequencies));
fLC = real(-((a * exp((-1i * pi) * (frequencies.' - frequencies))) / pi) .* sinc(a * (frequencies.' - frequencies) / pi));
fM = sparse([fO, -g * fI; fG, gain * zz]);

% Matrix that deletes the mean
Pi_nm = diag([0; ones(Nx-1, 1)]);

% Matrix that recovers only the mean
Pi_m = diag([1; zeros(Nx-1, 1)]);

% Modified fM
M_mod_nm = sparse([O, -g * I; G, gain * iDFT * Pi_nm * DFT * LC]);
fM_mod_nm = sparse([fO, -g * fI; fG, gain * Pi_nm * zz]);

% Vector with list of all the positive frequencies
pos_frequencies = frequencies(2:ceil((length(frequencies)-1)/2)+1);

% Matrix that deletes the mean and the high frequencies
Pi_lf = diag(abs(frequencies)<= N_ini_fq);
Pi_hf = diag(abs(frequencies)> N_ini_fq);

% Second Modified fM
M_mod_lf = sparse([O, -g * I; G, gain * iDFT * Pi_lf * Pi_nm * DFT * LC]);
fM_mod_lf = sparse([fO, -g * fI; fG, gain * Pi_lf * Pi_nm * zz]);

zz = iDFT*LC*DFT;
figure(1);
plot(frequencies,real(zz(1,:)),'x',frequencies,fLC(1,:),'o')

phi0 = ((abs(frequencies)<= N_ini_fq).*[0; ones(Nx-1, 1)].*rand(Nx,1)) + 1j * ((abs(frequencies)<= N_ini_fq).*[0; ones(Nx-1, 1)].*rand(Nx,1));
eta0 = 1i./g.*phi0.*frequencies;

eig_fM        = real(eig(full(fM)));
eig_fM_mod_nm = real(eig(full(fM_mod_nm)));
eig_fM_mod_lf = real(eig(full(fM_mod_lf)));
eig_PfM       = real(eig(full(kron(eye(2),Pi_nm * Pi_lf)*fM_mod_lf)));

conv_fM        = max(eig_fM(eig_fM<-1e-10));
conv_fM_mod_nm = max(eig_fM_mod_nm(eig_fM_mod_nm<-1e-10));
conv_fM_mod_lf = max(eig_fM_mod_lf(eig_fM_mod_lf<-1e-10));
conv_PfM       = max(eig_PfM(eig_PfM<-1e-10));


X0 = [phi0;eta0];

% [~, X_sol_fM] = ode45(@(t, y) fM*y, tspan, X0);
% [~, X_sol_fM_mod_nm] = ode45(@(t, y) fM_mod_nm*y, tspan, X0);
% [~, X_sol_fM_mod_lf] = ode45(@(t, y) fM_mod_lf*y, tspan, X0);

[~, X_sol_fM] = odeRK4_hom(fM, tspan, X0);
[~, X_sol_fM_mod_nm] = odeRK4_hom(fM_mod_nm, tspan, X0);
[~, X_sol_fM_mod_lf] = odeRK4_hom(fM_mod_lf, tspan, X0);

X_sol_fM = X_sol_fM';
X_sol_fM_mod_nm = X_sol_fM_mod_nm';
X_sol_fM_mod_lf = X_sol_fM_mod_lf';

[~, X_sol_M] = ode45(@(t, y) M*y, tspan, X0);
[~, X_sol_M_mod_nm] = ode45(@(t, y) M_mod_nm*y, tspan, kron(eye(2),iDFT)*X0);
[~, X_sol_M_mod_lf] = ode45(@(t, y) M_mod_lf*y, tspan, kron(eye(2),iDFT)*X0);

figure(2)
subplot(2,2,1)
semilogy(tspan,fHL_norm(X_sol_fM',frequencies),...
    tspan,fHL_norm(X_sol_fM_mod_nm',frequencies),...
    tspan,fHL_norm(X_sol_fM_mod_lf',frequencies),...
    tspan,exp(tspan.*conv_fM),...
    tspan,exp(tspan.*conv_fM_mod_nm),...
    tspan,exp(tspan.*conv_fM_mod_lf),...
    tspan,exp(tspan*conv_PfM))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,2)
semilogy(tspan,fHL_norm(kron(eye(2),Pi_nm * Pi_lf)*(X_sol_fM'),frequencies),...
    tspan,fHL_norm(kron(eye(2),Pi_nm * Pi_lf)*(X_sol_fM_mod_nm'),frequencies),...
    tspan,fHL_norm(kron(eye(2),Pi_nm * Pi_lf)*(X_sol_fM_mod_lf'),frequencies))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,3)
semilogy(tspan,fHL_norm(kron(eye(2),Pi_nm * Pi_hf)*(X_sol_fM'),frequencies),...
    tspan,fHL_norm(kron(eye(2),Pi_nm * Pi_hf)*(X_sol_fM_mod_nm'),frequencies),...
    tspan,fHL_norm(kron(eye(2),Pi_nm * Pi_hf)*(X_sol_fM_mod_lf'),frequencies))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,4)
semilogy(tspan,fHL_norm(kron(eye(2),Pi_m)*(X_sol_fM'),frequencies),...
    tspan,fHL_norm(kron(eye(2),Pi_m)*(X_sol_fM_mod_nm'),frequencies),...
    tspan,fHL_norm(kron(eye(2),Pi_m)*(X_sol_fM_mod_lf'),frequencies))
legend('fM','fM \0','fM \0 \hf')
grid()

figure(3)
subplot(2,2,1)
semilogy(tspan,HL_norm(X_sol_M',frequencies,Nx),...
    tspan,HL_norm(X_sol_M_mod_nm',frequencies,Nx),...
    tspan,HL_norm(X_sol_M_mod_lf',frequencies,Nx))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,2)
semilogy(tspan,HL_norm(kron(eye(2),iDFT * Pi_nm * Pi_lf * DFT)*(X_sol_M'),frequencies,Nx),...
    tspan,HL_norm(kron(eye(2),iDFT * Pi_nm * Pi_lf * DFT)*(X_sol_M_mod_nm'),frequencies,Nx),...
    tspan,HL_norm(kron(eye(2),iDFT * Pi_nm * Pi_lf * DFT)*(X_sol_M_mod_lf'),frequencies,Nx))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,3)
semilogy(tspan,HL_norm(kron(eye(2),iDFT * Pi_nm * Pi_hf * DFT)*(X_sol_M'),frequencies,Nx),...
    tspan,HL_norm(kron(eye(2),iDFT * Pi_nm * Pi_hf * DFT)*(X_sol_M_mod_nm'),frequencies,Nx),...
    tspan,HL_norm(kron(eye(2),iDFT * Pi_nm * Pi_hf * DFT)*(X_sol_M_mod_lf'),frequencies,Nx))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,4)
semilogy(tspan,HL_norm(kron(eye(2),iDFT * Pi_m * DFT)*(X_sol_M'),frequencies,Nx),...
    tspan,HL_norm(kron(eye(2),iDFT * Pi_m * DFT)*(X_sol_M_mod_nm'),frequencies,Nx),...
    tspan,HL_norm(kron(eye(2),iDFT * Pi_m * DFT)*(X_sol_M_mod_lf'),frequencies,Nx))
legend('fM','fM \0','fM \0 \hf')
grid()

figure(4)
subplot(2,2,1)
semilogy(tspan,vecnorm(X_sol_fM'),...
    tspan,vecnorm(X_sol_fM_mod_nm'),...
    tspan,vecnorm(X_sol_fM_mod_lf'))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,2)
semilogy(tspan,vecnorm(kron(eye(2),Pi_nm * Pi_lf)*(X_sol_fM')),...
    tspan,vecnorm(kron(eye(2),Pi_nm * Pi_lf)*(X_sol_fM_mod_nm')),...
    tspan,vecnorm(kron(eye(2),Pi_nm * Pi_lf)*(X_sol_fM_mod_lf')))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,3)
semilogy(tspan,vecnorm(kron(eye(2),Pi_nm * Pi_hf)*(X_sol_fM')),...
    tspan,vecnorm(kron(eye(2),Pi_nm * Pi_hf)*(X_sol_fM_mod_nm')),...
    tspan,vecnorm(kron(eye(2),Pi_nm * Pi_hf)*(X_sol_fM_mod_lf')))
legend('fM','fM \0','fM \0 \hf')
grid()
subplot(2,2,4)
semilogy(tspan,vecnorm(kron(eye(2),Pi_m)*(X_sol_fM')),...
    tspan,vecnorm(kron(eye(2),Pi_m)*(X_sol_fM_mod_nm')),...
    tspan,vecnorm(kron(eye(2),Pi_m)*(X_sol_fM_mod_lf')))
legend('fM','fM \0','fM \0 \hf')
grid()

figure(5)
semilogy(tspan,fHL_norm(X_sol_fM',frequencies),...
    'LineWidth',Linesize,'MarkerSize',Marksize)
legend('$[Err_1]$','Interpreter','latex','FontSize', Fontsize,'Location','northeast')
xlabel('time','Interpreter','latex','FontSize', Fontsize)
ylabel('norm $\|\cdot\|_{H^{1/2}\times L^2}$','Interpreter','latex','FontSize', Fontsize)
ylim([1e-9,1e1])
grid on
saveas(gcf,'figures/prez_lww_obs_conv_ini_sinus&cosinus.png')

figure(6)
semilogy(tspan,fHL_norm(X_sol_fM',frequencies),...
    tspan,fHL_norm(X_sol_fM_mod_nm',frequencies),...
    'LineWidth',Linesize,'MarkerSize',Marksize)
legend('$[Err_1]$','$[Err_2]$','Interpreter','latex','FontSize', Fontsize,'Location','northeast')
xlabel('time','Interpreter','latex','FontSize', Fontsize)
ylabel('norm $\|\cdot\|_{H^{1/2}\times L^2}$','Interpreter','latex','FontSize', Fontsize)
ylim([1e-9,1e1])
grid on
saveas(gcf,'figures/prez_lww_obs_conv_ini_sinus&cosinus_no_mean.png')

figure(7)
semilogy(tspan,fHL_norm(X_sol_fM',frequencies),...
    tspan,fHL_norm(X_sol_fM_mod_nm',frequencies),...
    tspan,fHL_norm(X_sol_fM_mod_lf',frequencies),...
    tspan,exp(tspan*conv_PfM),...
    'LineWidth',Linesize,'MarkerSize',Marksize)
legend('$[Err_1]$','$[Err_2]$','$[Err_3]$','$\exp(-t\max(\mathcal{R}e(\sigma(M_{[Err_3]}))))$','Interpreter','latex','FontSize', Fontsize,'Location','northeast')
xlabel('time','Interpreter','latex','FontSize', Fontsize)
ylabel('norm $\|\cdot\|_{H^{1/2}\times L^2}$','Interpreter','latex','FontSize', Fontsize)
ylim([1e-9,1e1])
grid on
saveas(gcf,'figures/prez_lww_obs_conv_ini_sinus&cosinus_no_mean_no_hf.png')

figure(8)
semilogy(tspan,fHL_norm(X_sol_fM',frequencies),...
    tspan,fHL_norm(X_sol_fM_mod_nm',frequencies),...
    tspan,fHL_norm(X_sol_fM_mod_lf',frequencies),...
    tspan,exp(tspan*conv_fM_mod_lf),...
    tspan,exp(tspan*conv_fM_mod_nm),...
    'LineWidth',Linesize,'MarkerSize',Marksize)
legend('$[Err_1]$','$[Err_2]$','$[Err_3]$','$\exp(-t\max(\mathcal{R}e(\sigma(M_{[Err_3]}))))$','$\exp(-t\max(\mathcal{R}e(\sigma(M_{[Err_2]}))))$','Interpreter','latex','FontSize', Fontsize,'Location','northeast')
xlabel('time','Interpreter','latex','FontSize', Fontsize)
ylabel('norm $\|\cdot\|_{H^{1/2}\times L^2}$','Interpreter','latex','FontSize', Fontsize)
ylim([1e-9,1e1])
grid on
saveas(gcf,'figures/prez_lww_obs_conv_ini_sinus&cosinus_no_mean_no_hf_2.png')

% ==================================================================================================================================
% ==================================================================================================================================
% Supporting functions
% ==================================================================================================================================
% ==================================================================================================================================

function norme = fHL_norm(vector,frequencies)
    f = abs(frequencies); f(1) = 1;
    tronc = size(vector,1)/2;
    phi = vector(1:tronc,:);
    eta = vector(tronc+1:end,:);
    norme = sum(f.*(abs(phi).^2)).^(1/2) + sum(abs(eta).^2).^(1/2);
end

% ===================================================

function norme = HL_norm(vector,frequencies,Nx)
    DFT = 1 / Nx * getDFT(Nx);
    f = abs(frequencies); f(1) = 1;
    tronc = size(vector,1)/2;
    phi = DFT * vector(1:tronc,:);
    eta = DFT * vector(tronc+1:end,:);
    norme = sum(f.*(abs(phi).^2)).^(1/2) + sum(abs(eta).^2).^(1/2);
end

% ===================================================

function DFT = getDFT(N)
    w_N = exp(-2i * pi / N);
    [rows, cols] = ndgrid(0:N-1, 0:N-1);
    DFT = w_N .^ (rows .* cols);
end

% ===================================================

function iDFT = getiDFT(N)
    w_N = exp(2i * pi / N);
    [rows, cols] = ndgrid(0:N-1, 0:N-1);
    iDFT = w_N .^ (rows .* cols);
end

% ===================================================

function A_dz = Matrix_A_dz(Nx,dx)
    k      = 2*pi*fftfreq(Nx, dx);
    kernel = real(ifft(abs(k)))*Nx;
    A_dz   = zeros(Nx);
    for i = 0:Nx-1
        A_dz(i+1,:) = (1/Nx)*circshift(kernel,i);
    end
end

% ===================================================

function freqs = fftfreq(Nx, dx)
    if mod(Nx, 2) == 0
        freqs = (-Nx/2:Nx/2-1)' / (Nx * dx);
    else
        freqs = (-(Nx-1)/2:(Nx-1)/2)' / (Nx * dx);
    end
    freqs = circshift(fftshift(freqs),1);
end

% ===================================================

function [nb_obs, C] = GetC(obs_map)
    C_flat = obs_map(:);
    nb_obs = sum(C_flat ~= 0);
    C = zeros(nb_obs, length(C_flat));
    j = 1;
    for p = 1:length(C_flat)
        if C_flat(p) ~= 0
            C(j, p) = C_flat(p);
            j = j + 1;
        end
    end
end

% ===================================================

function y = sinc(x)
    y = sin(pi * x) ./ (pi * x);
    y(x == 0) = 1;
end