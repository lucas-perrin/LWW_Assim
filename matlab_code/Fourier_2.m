Fontsize = 18;
Fontsize_label = 20;
Fontsize_axes = 18;
Linesize = 2;
Marksize = 9;

%% COMPUTATIONS
compute = 0;
movie   = 0;

if compute
    
clearvars -except Fontsize Fontsize_label Fontsize_axes Linesize Marksize compute movie

%   -----------------------
%-- Parameters to tune
%   -----------------------
x0       = 0;            % start of space domain
L        = 2*pi;         % length of spatial grid
Nx       = 2^8;        % number of points in space domain MUST BE ODD
Nf       = floor(Nx/2);  % number of frequences
dt       = 1e-3;         % timestep
a        = 1;            % value between 0 and pi
g        = 1;            % gravity constant set to 1
%g        = 9.80665;
gain     = 1;            % observer gain constant set to 1
T_start  = 0;            % start time
T        = 500;          % time window
N_ini_fq = 2^2; 

%   -----------------------
%-- Space discretization
%   -----------------------
dx         = L/(Nx);              % space steps size
Lx         = L - dx;              % L, for the frequency of the solution
xspan      = [x0:dx:(Lx+x0)];     % space grid
frequences = fftfreq(Nx,dx)*2*pi; % frequency space

%   -----------------------
%-- Time discretization
%   -----------------------
T_end   = T_start + T;    % end of time interval
tspan = T_start:dt:T_end; % 
Nt    = length(tspan);

%   -----------------------
%-- Matrix computations
%   -----------------------

% 1) DFT and iDFT Matrices (as well as no mean matrices)
DFT      = 1/Nx*getDFT(Nx);
iDFT     = getiDFT(Nx);

DFT_n_m  = 1/Nx*getDFT_no_mean(Nx);
iDFT_n_m = getiDFT_no_mean(Nx);

% 2) True matrix A not in Fourier
O = zeros(Nx,Nx);
I = eye(Nx);
G  = Matrix_A_dz(Nx,dx);
[~,C] = GetC(abs(xspan - pi) <= a);
obs_space = abs(xspan - pi) <= a;
C = - C' * C;
A = [O, - I; G, C];

% 2) True matrix A in Fourier
% fO =  zeros(2*Nf+1,2*Nf+1);
% fI =  - eye(2*Nf+1);
% fF = diag(abs([[0:1:Nf],[-Nf:1:-1]]));
% fC = real(-((a*exp((-1i*pi) .* (frequences'-frequences))) ./ pi) .* sinc(a*(frequences'-frequences)/(pi)));
% fA = [fO,fI;fF,fC];

% 3) computing the pseudo matrix A (size (Nx-1)^2) :

% getting A in Fourier : DFT * A * iDFT :
A_f = kron(eye(2),DFT) * A * kron(eye(2),iDFT);

% removing the mean columns and rows :
A_f_11 = A_f(1:Nx,1:Nx);
A_f_12 = A_f(1:Nx,Nx+1:2*Nx);
A_f_21 = A_f(Nx+1:2*Nx,1:Nx);
A_f_22 = A_f(Nx+1:2*Nx,Nx+1:2*Nx);
A_fr   = [A_f_11(2:Nx,2:Nx), A_f_12(2:Nx,2:Nx) ; A_f_21(2:Nx,2:Nx) , A_f_22(2:Nx,2:Nx)];

% getting back in space with a pseudo DFT :
A_fr_s = kron(eye(2),iDFT(1:Nx,2:Nx)) * A_fr * kron(eye(2),DFT(2:Nx,1:Nx));
A_fr_s_2 = kron(eye(2),iDFT(2:Nx,2:Nx)) * A_fr * kron(eye(2),DFT(2:Nx,2:Nx));


% obtaining the convergence factor for this matrix :
eig_A_fr_s     = eig(A_fr_s);
eig_A_fr_s     = eig_A_fr_s(abs(eig(A_fr_s)) > 1e-10);
conv_fact_fr_s = min(abs(eig_A_fr_s));

% eigenvalues of A :
[V,D] = eig(A);
eig_A = diag(D);
cfactor = min(abs(real(eig_A(abs(imag(eig_A)) > 1e-10))));
place_cfactor = find(abs(real(eig_A)) == min(abs(real(eig_A(abs(imag(eig_A)) > 1e-10)))));
eig_cfactor = eig_A(place_cfactor);

% Projection on N_ini_fq :
DFh = del_hig_freqs(frequences,N_ini_fq);
A_ini_f = kron(eye(2),DFh*DFT) * A * kron(eye(2),iDFT*DFh.');
A_ini_s = kron(eye(2),DFh*iDFT) * A * kron(eye(2),DFT*DFh.');

% eigenvalues of A_ini_s :
[V_ini_s,D_ini_s] = eig(A_ini_s);
eig_A_ini_s = diag(D_ini_s);
cfactor = min(abs(real(eig_A_ini_s(abs(imag(eig_A_ini_s)) > 1e-10))));
place_cfactor_ini_s = find(abs(real(eig_A_ini_s)) == min(abs(real(eig_A_ini_s(abs(imag(eig_A_ini_s)) > 1e-10)))));
eig_cfactor_ini_s = eig_A_ini_s(place_cfactor_ini_s);

end

% plot the eigenvalues
figure(1)
subplot(1,2,1)
plot(real(eig(A_ini_f)),imag(eig(A_ini_f)),'x',...
    real(eig(A_fr)),imag(eig(A_fr)),'o','MarkerSize',Marksize,'LineWidth',Linesize)
legend('A f','A fr')
grid on
title('Fourier')
subplot(1,2,2)
plot(real(eig(A)),imag(eig(A)),'x',...
    real(eig(A_fr_s)),imag(eig(A_fr_s)),'o',...
    real(eig(A_fr_s_2)),imag(eig(A_fr_s_2)),'s',...
    real(eig(A_ini_s)),imag(eig(A_ini_s)),'^',...
    real(eig_cfactor),imag(eig_cfactor),'d','MarkerSize',Marksize,'LineWidth',Linesize)
legend('A','A fr s','A fr s 2')
grid on
% xlim([-2,0])
% ylim([-1,1])
title('Space')

if compute
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
e0 = [f1,f2];

%   -----------------------
%-- Solver
%   -----------------------
G_zero = @(t) zeros(2*Nx,1);

fprintf('...solving error equation... \n')
[~,solve_error] = odeRK4_inhom_ufunc(sparse(A),G_zero,tspan,e0);
fprintf('-> done \n')

fprintf('...solving error equation one step... \n')
[~,solve_error_os] = odeRK4_inhom_ufunc(sparse(A),G_zero,tspan(1:2),e0);
fprintf('-> done \n')

e0_2 = kron(eye(2),DFT*diag(abs(frequences) > N_ini_fq).*(abs(frequences)>0)*iDFT)*solve_error_os(:,end);

% error
phi_error = solve_error(1:Nx,:);
eta_error = solve_error(Nx+1:2*Nx,:);

% vecnorm error
Pi_BF = iDFT*(DFh.'*DFh)*DFT;
DFl = del_low_freqs(frequences,N_ini_fq);
Pi_HF = iDFT*(DFl.'*DFl)*DFT;

vec_phi_error = vecnorm(phi_error,2);
vec_eta_error = vecnorm(eta_error,2);
vec_error     = vecnorm(solve_error,2);

vec_phi_error_lf = vecnorm(Pi_BF*phi_error,2);
vec_eta_error_lf = vecnorm(Pi_BF*eta_error,2);
vec_error_lf     = vecnorm(kron(eye(2),Pi_BF)*solve_error,2);

vec_phi_error_hf = vecnorm(Pi_HF * phi_error,2);
vec_eta_error_hf = vecnorm(Pi_HF * eta_error,2);
vec_error_hf     = vecnorm(kron(eye(2),Pi_HF)*solve_error,2);

convergence   = cond(V).*norm(e0_2,2).*exp(-tspan.*abs(real(eig_cfactor(1))));
convergence2  = cond(V_ini_s).*norm(e0,2).*exp(-tspan.*abs(real(eig_cfactor_ini_s(1))));

%   -----------------------
%-- Solver Projected equation
%   -----------------------

A_plf = [O, - I; G, iDFT*DFh.'*DFh*DFT*C];

fprintf('...solving projected LF error equation... \n')
[~,solve_error_plf] = odeRK4_inhom_ufunc(sparse(A_plf),G_zero,tspan,e0);
fprintf('-> done \n')

phi_error_plf = solve_error_plf(1:Nx,:);
eta_error_plf = solve_error_plf(Nx+1:2*Nx,:);

vec_phi_error_plf = vecnorm(phi_error_plf,2);
vec_eta_error_plf = vecnorm(eta_error_plf,2);
vec_error_plf     = vecnorm(solve_error_plf,2);

%nb_points = floor(100 * T/50);
nb_points = 100;
points_list = 1:floor(length(tspan)/nb_points):length(tspan);

end

figure(3)
subplot(1,3,1)
semilogy(tspan(points_list),vec_phi_error(points_list),'-x',...
    tspan(points_list),vec_eta_error(points_list),'-o',...
    tspan(points_list),vec_error(points_list),'-d',...
    tspan(points_list),convergence(points_list),'-',...
    tspan(points_list),convergence2(points_list),'-','MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
ylim([1e-15,1e2])
title(['$N_x =$ ',num2str(Nx),', $N_{ini} =$ ',num2str(N_ini_fq)],'Interpreter','Latex','FontSize', Fontsize_label)
xlabel('time $t$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('error in $L^2$ norm','Interpreter','Latex','FontSize', Fontsize_label)
grid on
legend('$\left\|\hat{\varphi_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{\eta_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{x}\right\|_{2} = \left\|(\hat{\varphi_\varepsilon},\hat{\eta_\varepsilon})^t\right\|_{2}$',...
    'conv. fac. high freq.',...
    'conv. fac. cond. ini.',...
    'Interpreter','latex','Fontsize', Fontsize)
subplot(1,3,2)
semilogy(tspan(points_list),vec_phi_error_lf(points_list),'-x',...
    tspan(points_list),vec_eta_error_lf(points_list),'-o',...
    tspan(points_list),vec_error_lf(points_list),'-d',...
    tspan(points_list),convergence(points_list),'-',...
    tspan(points_list),convergence2(points_list),'-','MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
ylim([1e-15,1e2])
title(['$N_x =$ ',num2str(Nx),', $N_{ini} =$ ',num2str(N_ini_fq)],'Interpreter','Latex','FontSize', Fontsize_label)
xlabel('time $t$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('error in $L^2$ norm','Interpreter','Latex','FontSize', Fontsize_label)
grid on
legend('$\left\|\hat{\varphi_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{\eta_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{x}\right\|_{2} = \left\|(\hat{\varphi_\varepsilon},\hat{\eta_\varepsilon})^t\right\|_{2}$',...
    'conv. fac. high freq.',...
    'conv. fac. cond. ini.',...
    'Interpreter','latex','Fontsize', Fontsize)
subplot(1,3,3)
semilogy(tspan(points_list),vec_phi_error_hf(points_list),'-x',...
    tspan(points_list),vec_eta_error_hf(points_list),'-o',...
    tspan(points_list),vec_error_hf(points_list),'-d',...
    tspan(points_list),convergence(points_list),'-',...
    tspan(points_list),convergence2(points_list),'-','MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
ylim([1e-15,1e2])
title(['$N_x =$ ',num2str(Nx),', $N_{ini} =$ ',num2str(N_ini_fq)],'Interpreter','Latex','FontSize', Fontsize_label)
xlabel('time $t$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('error in $L^2$ norm','Interpreter','Latex','FontSize', Fontsize_label)
grid on
legend('$\left\|\hat{\varphi_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{\eta_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{x}\right\|_{2} = \left\|(\hat{\varphi_\varepsilon},\hat{\eta_\varepsilon})^t\right\|_{2}$',...
    'conv. fac. high freq.',...
    'conv. fac. cond. ini.',...
    'Interpreter','latex','Fontsize', Fontsize)
saveas(gcf,['coude_Nx',num2str(Nx),'_Ni',num2str(N_ini_fq),'.png'])

figure(4)
subplot(1,2,1)
semilogy(tspan(points_list),vec_phi_error_hf(points_list),'-x',...
    tspan(points_list),vec_phi_error_lf(points_list),'-o',...
    tspan(points_list),vec_phi_error(points_list),'-d',...
    tspan(points_list),convergence(points_list),'-',...
    tspan(points_list),convergence2(points_list),'-','MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
ylim([1e-15,1e2])
title(['$N_x =$ ',num2str(Nx),', $N_{ini} =$ ',num2str(N_ini_fq)],'Interpreter','Latex','FontSize', Fontsize_label)
xlabel('time $t$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('error in $L^2$ norm','Interpreter','Latex','FontSize', Fontsize_label)
grid on
legend('hf',...
    'lf',...
    'af',...
    'conv. fac. high freq.',...
    'conv. fac. cond. ini.',...
    'Interpreter','latex','Fontsize', Fontsize)
subplot(1,2,2)
semilogy(tspan(points_list),vec_eta_error_hf(points_list),'-x',...
    tspan(points_list),vec_eta_error_lf(points_list),'-o',...
    tspan(points_list),vec_eta_error(points_list),'-d',...
    tspan(points_list),convergence(points_list),'-',...
    tspan(points_list),convergence2(points_list),'-','MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
ylim([1e-15,1e2])
title(['$N_x =$ ',num2str(Nx),', $N_{ini} =$ ',num2str(N_ini_fq)],'Interpreter','Latex','FontSize', Fontsize_label)
xlabel('time $t$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('error in $L^2$ norm','Interpreter','Latex','FontSize', Fontsize_label)
grid on
legend('hf',...
    'lf',...
    'af',...
    'conv. fac. high freq.',...
    'conv. fac. cond. ini.',...
    'Interpreter','latex','Fontsize', Fontsize)

figure(5)
semilogy(tspan(points_list),vec_phi_error_plf(points_list),'-x',...
    tspan(points_list),vec_eta_error_plf(points_list),'-o',...
    tspan(points_list),vec_error_plf(points_list),'-d',...
    tspan(points_list),convergence2(points_list),'-','MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
ylim([1e-15,1e2])
%title('solving $\partial_t^2\Pi_L \varphi_\varepsilon + (-\Delta)^2\Pi_L \varphi_\varepsilon + \Pi_L\mathbf{1}_{-a\leq x \leq a}\partial_t \Pi_L \varphi_\varepsilon = 0$','Interpreter','Latex','FontSize', Fontsize_label)
title('solving $\partial_t^2 \varphi_\varepsilon + (-\Delta)^2 \varphi_\varepsilon + \Pi_L\mathbf{1}_{-a\leq x \leq a}\partial_t \varphi_\varepsilon = 0$','Interpreter','Latex','FontSize', Fontsize_label)
subtitle(['$N_x =$ ',num2str(Nx),', $N_{ini} =$ ',num2str(N_ini_fq)],'Interpreter','Latex','FontSize', Fontsize_label)
xlabel('time $t$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('error in $L^2$ norm','Interpreter','Latex','FontSize', Fontsize_label)
grid on
legend('$\left\|\hat{\varphi_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{\eta_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{x}_\varepsilon\right\|_{2} = \left\|(\hat{\varphi_\varepsilon},\hat{\eta_\varepsilon})^t\right\|_{2}$',...
    'conv. fac.',...
    'Interpreter','latex','Fontsize', Fontsize)
saveas(gcf,['proj_low_freq_coude_Nx',num2str(Nx),'_Ni',num2str(N_ini_fq),'.png'])

figure(6)
semilogy(tspan(points_list),vec_phi_error(points_list),'-x',...
    tspan(points_list),vec_eta_error(points_list),'-o',...
    tspan(points_list),vec_error(points_list),'-d',...
    'MarkerSize',Marksize,'LineWidth',Linesize)
set(gca,'FontSize',Fontsize_axes)
ylim([1e-5,1e2])
title('solving $\partial_t^2 \varphi_\varepsilon + (-\Delta)^2 \varphi_\varepsilon + \mathbf{1}_{-a\leq x \leq a}\partial_t \varphi_\varepsilon = 0$','Interpreter','Latex','FontSize', Fontsize_label)
subtitle(['$N_x =$ ',num2str(Nx),', $N_{ini} =$ ',num2str(N_ini_fq)],'Interpreter','Latex','FontSize', Fontsize_label)
xlabel('time $t$','Interpreter','Latex','FontSize', Fontsize_label)
ylabel('error in $L^2$ norm','Interpreter','Latex','FontSize', Fontsize_label)
grid on
legend('$\left\|\hat{\varphi_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{\eta_\varepsilon}\right\|_{2}$',...
    '$\left\|\hat{x}_\varepsilon\right\|_{2} = \left\|(\hat{\varphi_\varepsilon},\hat{\eta_\varepsilon})^t\right\|_{2}$',...
    'Interpreter','latex','Fontsize', Fontsize)
saveas(gcf,['no_conv_coude_Nx',num2str(Nx),'_Ni',num2str(N_ini_fq),'.png'])

if movie
    
change = [zeros(Nx/2,Nx/2),eye(Nx/2);eye(Nx/2),zeros(Nx/2,Nx/2)];

for j = 1:nb_points
figure(5)
subplot(2,1,1)
semilogy((change*frequences.'),(change*abs(DFT*phi_error(:,points_list(j)))))
ylim([1e-16,1e1])
subplot(2,1,2)
semilogy(change*frequences.',change*abs(DFT*eta_error(:,points_list(j))))
ylim([1e-16,1e1])
pause(0.1)
end

end

%% functions

function DF = del_hig_freqs(frequences,thrs)
DF = diag(abs(frequences) <= thrs).*(abs(frequences)>0);
k = 1;
while k <= size(DF,1)
    if sum(DF(k,:)) == 0
        DF(k,:) = [];
    else
        k = k+1;
    end
end
end

function DF = del_low_freqs(frequences,thrs)
DF = diag(abs(frequences) > thrs).*(abs(frequences)>0);
k = 1;
while k <= size(DF,1)
    if sum(DF(k,:)) == 0
        DF(k,:) = [];
    else
        k = k+1;
    end
end
end

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