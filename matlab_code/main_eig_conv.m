clear all
close all

% graphics settings
Fontsize = 18;
Fontsize_label = 22;
Fontsize_axes = 18;
Linesize = 2;
Marksize = 9;

a = 3/2;
L = 2 * pi;
g = 9.80665;

Nx_list = ceil(2.^[2:0.2:12]+1);
mod_Nx_list = mod(Nx_list,2);
Nx_list = Nx_list + 1 - mod_Nx_list;
Nf_list = floor(Nx_list / 2);

conv_fact = zeros(1,size(Nx_list,2));

for Nx_i = 1:size(Nx_list,2)
    fprintf('doing %i over %i ... \n',Nx_i,size(Nx_list,2))
    Nx = Nx_list(Nx_i);
    fM = get_fM(L,Nx,a,3);
    eig_fM  = real(eig(full(fM)));
    conv_fact(Nx_i) = max(eig_fM(eig_fM<0));
end

figure(1)
loglog(Nf_list,conv_fact,'-x',...
    Nf_list,-exp(-Nf_list.^(1/4)),'-',...
    Nf_list,-(Nf_list).^(-1/2),'-','LineWidth',Linesize,'MarkerSize',Marksize)
set(gca,'FontSize',Fontsize_axes)
legend('$\max(\mathcal{R}e(\sigma(M))<0)$','$-\exp(-N_f^{1/4})$','$-N_f^{-1/2}$','Interpreter','latex','FontSize', Fontsize,'Location','northwest')
xlabel('$N_f$ : number of frequencies','Interpreter', 'latex','FontSize', Fontsize_label)
ylabel('convergence rate','Interpreter', 'latex','FontSize', Fontsize_label)
grid on

% ==================================================================================================================================
% ==================================================================================================================================
% Supporting functions
% ==================================================================================================================================
% ==================================================================================================================================

function fM = get_fM(L,Nx,a,d)
    g = 9.80665;
    gain = 10;
    Nf = floor(Nx / 2);
    dx = L / Nx;
    Lx = L - dx;
    xspan = 0:dx:Lx;
    frequencies = fftfreq(Nx, dx)*2*pi;
    fO = zeros(2 * Nf + 1, 2 * Nf + 1);
    fI = eye(2 * Nf + 1);
    fG = diag(abs(frequencies));
    fLC = real(-((a * exp((-1i * pi) * (frequencies.' - frequencies))) / pi) .* sinc(a * (frequencies.' - frequencies) / pi));
    fM = sparse([fO, -g * fI; fG, gain * fLC]);
end

% ===================================================

function norme = HL_norm(vector,frequencies)
    f = abs(frequencies); f(1) = 1;
    tronc = size(vector,1)/2;
    phi = vector(1:tronc,:);
    eta = vector(tronc+1:end,:);
    norme = sum(f.*(abs(phi).^2)).^(1/2) + sum(abs(phi).^2).^(1/2);
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