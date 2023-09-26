clear all
close all

Nx_list  = 2:2:500;
size_obs = 0.5;       % value between 0 and 1
x0       = 0;         % start of space domain
Lx       = 1;

for j = 1:length(Nx_list)
    Nx           = Nx_list(j);
    dx           = Lx/(Nx - 1);        % space steps size
    xspan        = [x0:dx:Lx];          % space grid
    
    pn_wout_zero = Proj_wout_zero(Nx,dx);
    PN_wout_zero = [pn_wout_zero, zeros(Nx); zeros(Nx), pn_wout_zero];
    A_dz         = Matrix_A_dz(Nx,dx);
    A            = [zeros(Nx), -eye(Nx); A_dz, zeros(Nx)];
    obs_map      = abs(xspan) <= size_obs * Lx;
    [m_obs,Cc]   = GetC(obs_map);
    C            = [zeros(m_obs,Nx) Cc];
    
    M = A - PN_wout_zero * (C') * C;
    %M = A - (C') * C;
    
    conv(j) = min(abs(eig(M)));
end

figure(1)
spy(M)

figure(2)
semilogy(Nx_list,conv)

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