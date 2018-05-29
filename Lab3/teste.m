clc;

% frequências desejadas (em BPM)
fd1_bpm = 150;
fd2_bpm = 50;

%frequências desejadas em Hz
fprintf('Frequency 1 to find');
fd1 = (fd1_bpm/60)/2
fprintf('Frequency 2 to find');
fd2 = (fd2_bpm/60)/2

% int_ms = 0.464 - 0.09285; 
% ms = 0.09285:int_ms/50:0.464;
ms = linspace(0.09285, 0.464, 100);

% int_l = 0.25 - 0.05;
% l = 0.05:int_l/50:0.25;
l = linspace(0.05, 0.25, 100);

fd_aux = zeros([100 100]);
wn_mtr = zeros([100 100]);

beta = 0.001;
k=0.35;
M=0.1;
L=0.25;

for i=1:100
    for j = 1:100
        [wn, dt] = calc_wn(beta, k, l(i), ms(j), L, M);
        wn_mtr(i,j) = wn;
        %wd - frequência de oscilação amortecida 
        if imag(wn) == 0 && (1-dt^2)>0
            fd_aux(i,j) = (1/(2*pi))*wn*sqrt(1-dt^2);
        end
    end
end

% Por inspecção podemos encontrar um m e dois l que satisfaçam 
figure(1);
plot3(l, ms, fd_aux);

figure(2);
plot(l, fd_aux(:,1));

%Get a value of m 
index_l1 = 1;
index_l2 = 1;
exist1 = 0;
exist2 = 0;
err1 = 10;
err2 = 10;
min_err1 =10;
min_err2 =10;
mass_idx = 1;

for i=100:-1:1
    for j=100:-1:1
        err1 = abs(fd_aux(j,i) - fd1);
        if err1 <= 0.005 && err1 <= min_err1
            index_l1 = j;
            min_err1 = err1;
        end
        err2 = abs(fd_aux(j,i) - fd2);
        if err2 <= 0.005 && err2 <= min_err2
            index_l2 = j;
            min_err2 = err2;
        end
    end
      if min_err1 ~=10 && min_err2 ~=10
          mass_idx = i;
      end
      min_err1 = 10;
      min_err2 = 10;
end

mass_idx
fprintf('Valor de massa');
ms(mass_idx)
fprintf('Valor de l1')
l(index_l1)
fprintf('Valor de l2')
l(index_l2)

figure(3);
plot(l,fd_aux(:,mass_idx));


