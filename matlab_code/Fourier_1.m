clear all
close all

Fontsize = 18;
Fontsize_label = 28;
Fontsize_axes = 18;
Linesize = 2;
Marksize = 9;

%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;            % start of space domain
L        = 2*pi;         % length of spatial grid
Nx       = 2^2+1;        % number of points in space domain MUST BE ODD
Nf       = floor(Nx/2);  % number of frequences
dt       = 1e-3;         % timestep
size_obs = 0.5;          % value between 0 and 1
g        = 1;            % gravity constant set to 1
gain     = 1;            % gain constant set to 1
T_start  = 0;            % start time
T        = 250;          % time window
a        = 1;

%   -----------------------
%-- Space discretization
%   -----------------------
dx    = L/(Nx);        % space steps size
Lx     = L - dx;       % L, for the frequency of the solution
xspan = [x0+dx/2:dx:(Lx+x0)+dx/2]; % space grid
xspan = [x0:dx:(Lx+x0)]; % space grid

frequences = fftfreq(Nx,dx)*2*pi;

sinx = sin(xspan);

fftsinx = fft(sinx,Nx)/(Nx);

figure(1)
plot(xspan, sinx)

figure(2)
plot(real(fftsinx),imag(fftsinx),'o')
xlim([-10,10])
grid on

fftfreq(Nx,dx);

%Nf = 2;

% signal in Fourier :
ak = zeros(1,Nf); % amplitudes of the cosinus of frequences 1,2,...
bk = zeros(1,Nf); % amplitudes of the sinus of frequences 1,2,...
bk(1) = 1;
cp = (ak - 1i.*bk)/2;
cm = (ak + 1i.*bk)/2;
ck = [0, cp, cm(end:-1:1)];

DFT = 1/Nx*getDFT(Nx);
iDFT = getiDFT(Nx);

DFT_n_m = 1/Nx*getDFT_no_mean(Nx);
iDFT_n_m = getiDFT_no_mean(Nx);

% check if the formulas for the DFT and iDFT works :
fprintf('check formula iDFT : err = %d \n',norm(iDFT*ck.' - sinx.',2))
fprintf('check formula DFT : err = %d \n',norm(DFT*sinx.' - ck.',2))

v = (1/(2*pi)) * sinc(([0:2*Nf-1])/pi);
C = toeplitz([v(1) fliplr(v(2:end))], v);

F = diag(abs([[-Nf:-1],[1:Nf]]));
I = eye(2*Nf);
O = zeros(2*Nf,2*Nf);
A = [O,I;F,C];

O2 =  zeros(2*Nf+1,2*Nf+1);
I2 =  - eye(2*Nf+1);
F2 = diag(abs([[0:1:Nf],[-Nf:1:-1]]));
C2 = real(-((a*exp((-1i*pi) .* (frequences'-frequences))) ./ pi) .* sinc(a*(frequences'-frequences)/(pi)));
A2 = [O2,I2;F2,C2];

O3 = zeros(Nx,Nx);
I3 = - eye(Nx);
G  = Matrix_A_dz(Nx,dx);
[~,C3] = GetC(abs(xspan - pi) <= a);
obs_space = abs(xspan - pi) <= a;
C3 = - C3' * C3;
A3 = [O3, I3; G, C3];

A3F = kron(eye(2),DFT)*A3*kron(eye(2),iDFT);
Ctest = real(A3F(Nx+1:end,Nx+1:end));

figure(3)
plot(frequences,Ctest(1,:),'-o',...
    frequences,C2(1,:),'-x')
grid on
legend('true','Fourier')
fprintf('error on Fourier of indicatrice %d \n',norm(Ctest(1,:) - C2(1,:),2))

O_ = zeros(Nx,Nx);
I_ = eye(Nx);
G_ = Matrix_A_dz(Nx,dx);
[~,C_] = GetC(abs(xspan) <= 1);
obs_space = abs(xspan) <= 1;
C_ = - C_' * C_;
A_ = [O_, -I_; G_, C_];

% Matrix A in Fourier
A_f = kron(eye(2),DFT)*A_*kron(eye(2),iDFT);
fprintf('check real/imag A_f : diff = %d \n',norm(A_f - real(A_f)));
A_f = real(A_f);

% Matrix A modified : in Fourier, no mean
A_f_m = kron(eye(2),DFT_n_m)*A_*kron(eye(2),iDFT_n_m);
fprintf('check real/imag A_f_m : diff = %d \n',norm(A_f_m - real(A_f_m)));
A_f_m = real(A_f_m);

% Matrix A in Fourier, reduced : removing mode 0
A_f_11 = A_f(1:Nx,1:Nx);
A_f_12 = A_f(1:Nx,Nx+1:2*Nx);
A_f_21 = A_f(Nx+1:2*Nx,1:Nx);
A_f_22 = A_f(Nx+1:2*Nx,Nx+1:2*Nx);
A_f_r  = [A_f_11(2:Nx,2:Nx), A_f_12(2:Nx,2:Nx) ; A_f_21(2:Nx,2:Nx) , A_f_22(2:Nx,2:Nx)];

% Matrix A Fourier reduced, way back :
A_f_r_b = kron(eye(2),iDFT(2:Nx,2:Nx))*A_f_r*kron(eye(2),DFT(2:Nx,2:Nx));
fprintf('check real/imag A_f_r_b : diff = %d \n',norm(A_f_r_b - real(A_f_r_b)));
A_f_r_b = real(A_f_r_b);

% Matrix A modified in Fourier, way back
A_f_m_b = kron(eye(2),iDFT_n_m)*A_f_m*kron(eye(2),DFT_n_m);
fprintf('check real/imag A_f_m_b : diff = %d \n',norm(A_f_m_b - real(A_f_m_b)));
A_f_m_b = real(A_f_m_b);

% figure(4)
% A2F = kron(eye(2),iDFT)*A2*kron(eye(2),DFT);
% spy(A2F)

%%

if 0
Nx_list = floor(10.^(linspace(1,3,10)));
conv_fact = zeros(1,length(Nx));

for k = 1:length(Nx_list)
    Nx = Nx_list(k);
    
    x0       = -pi;         % start of space domain
    L        = 2*pi;            % length of spatial grid
    
    dx    = L/(Nx);        % space steps size
    Lx     = L - dx;       % L, for the frequency of the solution
    xspan = [x0:dx:(Lx+x0)]; % space grid
    
    % génération de la matrice A
    O = zeros(Nx,Nx);
    I = eye(Nx);
    G  = Matrix_A_dz(Nx,dx);
    [~,C] = GetC(abs(xspan) <= 1);
    obs_space = abs(xspan) <= 1;
    C = - C' * C;
    A = [O, - I; G, C];
    
    % matrices de passage en Fourier
    DFT = 1/Nx*getDFT(Nx);
    iDFT = getiDFT(Nx);
    DFT_n_m = 1/Nx*getDFT_no_mean(Nx);
    iDFT_n_m = getiDFT_no_mean(Nx);
    
    % matrice A_F = matrice A en Fourier :
    A_F = kron(eye(2),DFT)*A*kron(eye(2),iDFT);
    
    % matrice A_F_R : matrice A_F réduite, sans le mode 0 :
    A_F_11 = A_F(1:Nx,1:Nx);
    A_F_12 = A_F(1:Nx,Nx+1:2*Nx);
    A_F_21 = A_F(Nx+1:2*Nx,1:Nx);
    A_F_22 = A_F(Nx+1:2*Nx,Nx+1:2*Nx);
    A_F_R   = [A_F_11(2:Nx,2:Nx), A_F_12(2:Nx,2:Nx) ; A_F_21(2:Nx,2:Nx) , A_F_22(2:Nx,2:Nx)];
    
    % nouvelle matrice A_F_R_B : A_F_R en non-Fourier :
    A_F_R_B    = kron(eye(2),iDFT(2:Nx,2:Nx))*A_F_R*kron(eye(2),DFT(2:Nx,2:Nx));
    
    % matrice A_F_M_B : matrice A avec passage en Fourier et retour en
    % non-Fourier avec les matrices de DFT et iDFT sans la moyenne :
    A_F_M = kron(eye(2),DFT_n_m)*A*kron(eye(2),iDFT_n_m);
    A_F_M_B = kron(eye(2),iDFT_n_m)*A_F_M*kron(eye(2),DFT_n_m);
    
    % valeurs propres de A_F_R_B :
    eigenval_F_R_B     = eig(A_F_R_B);
    eigenval_F_R_B     = eigenval_F_R_B(abs(eig(A_F_R_B)) > 1e-10);
    conv_fact_F_R_B(k) = min(abs(eigenval_F_R_B));
    
    % valeurs propres de A_F_M_B :
    eigenval_F_M_B     = eig(A_F_M_B);
    eigenval_F_M_B     = eigenval_F_M_B(abs(eig(A_F_M_B)) > 1e-10);
    conv_fact_F_M_B(k) = min(abs(eigenval_F_M_B)); 
    
end

figure(5)
semilogy(Nx_list,conv_fact_F_R_B,'-o',...
    Nx_list,conv_fact_F_M_B,'-x')
legend('A_F_R_B','A_F_M_B')
xlabel('points on the mesh : $N_x$','Interpreter','Latex','FontSize', Fontsize)
ylabel('speed of convergence : $\min |\lambda|$','Interpreter','Latex','FontSize', Fontsize)
grid on
end

%%

if 1
max_dim = 10;
Nx_list = floor(2.^[3:0.5:max_dim]) + mod(floor(2.^[3:0.5:max_dim]),2);
conv_fact = zeros(1,length(Nx_list));

for k = 1:length(Nx_list)
    Nx = Nx_list(k);
    
    x0    = 0;         % start of space domain
    L     = 2*pi;            % length of spatial grid
    a     = 1;
    dx    = L/(Nx);        % space steps size
    Lx    = L - dx;       % L, for the frequency of the solution
    xspan = [x0:dx:(Lx+x0)]; % space grid
    
    % Matrix A :
    O = zeros(Nx,Nx);
    I = eye(Nx);
    G  = Matrix_A_dz(Nx,dx);
    [~,C] = GetC(abs(xspan - pi) <= a);
    obs_space = abs(xspan - pi) <= a;
    C = - C' * C;
    A = [O, - I; G, C];
    
    % eigenvalues of A :
    eig_A = eig(A);
    cfactor = min(abs(real(eig_A(abs(imag(eig_A)) > 1e-1))));
    place_cfactor = find(abs(real(eig_A)) == min(abs(real(eig_A(abs(imag(eig_A)) > 1e-1)))));
    eig_cfactor = eig_A(place_cfactor);
    conv_fact(k) = abs(real(eig_cfactor(1)));
end

figure(6)
loglog(Nx_list,conv_fact,'x-',...
    'MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
title('Convergence factor vs. dimension (even)','Interpreter','Latex','FontSize',Fontsize_label)
xlabel('points on the mesh : $N_x$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('convergence factor','Interpreter','Latex','FontSize', Fontsize_label)
grid on
saveas(gcf,'convergence_factor_even.png')
end
%% functions

function k = fftfreq(n,d)
    % f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
    % f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    if mod(n,2)
        k = [[0:(n-1)/2],[-(n-1)/2:-1]]./(d*(n));
    else
        k = [[0:n/2-1],[-n/2:-1]]./(d*(n));
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

function W_N = getDFT(N)
w_N = exp(-2*1i*pi/N);
W_N = w_N.^([0:N-1]'*[0:N-1]);
end

function W_N = getiDFT(N)
w_N = exp(+2*1i*pi/N);
W_N = w_N.^([0:N-1]'*[0:N-1]);
end

function W_N = getDFT_no_mean(N)
w_N = exp(-2*1i*pi/N);
W_N = w_N.^([0:N-1]'*[0:N-1]);
W_N = W_N(2:end,1:end);
end

function W_N = getiDFT_no_mean(N)
w_N = exp(+2*1i*pi/N);
W_N = w_N.^([0:N-1]'*[0:N-1]);
W_N = W_N(1:end,2:end);
end

function W_N = getW_2(N)
w_N = exp(-1i/N);
W_N = w_N.^([0:N-1]'*[0:N-1]);
end