clear all
close all

x = linspace(0,30,30*1e3);
y = x + 2*randn(1,length(x));

figure(1)
plot(x,y,'o')


reglin_1 = y'\x';

figure(2)
plot(x,y,'o',...
    x,reglin_1.*x,'-')
title(['y = $\beta x$,  $\beta$ = ',num2str(reglin_1)], 'Interpreter', 'latex')

%x = linspace(0,100,100);
y = exp(-x + 2*randn(1,length(x)) - 10);

figure(3)
semilogy(x,y,'o')

reglin_2 = [ones(length(x),1) log(y')]\x';
reglin_2 = [a,b] = polyfit(x,log(y),1);
figure(4)
semilogy(x,y,'o',...
    x,exp(reglin_2(2).*x),'-')
title(['y = $e^{\beta x}$,  $\beta$ = ',num2str(reglin_2(2))], 'Interpreter', 'latex')