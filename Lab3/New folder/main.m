%% Project by Dinis Rodrigues nº79089 and José Fernandes nº82414
%  For the 1st Laboratory of MSIM     


%% Ex5
% 
function main

close all
file='ex5';
load_system('ex5');
gamma1=0.1175;
gamma2=0.0445;
g=9.8;
k=3;
beta=0.1;
param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(beta/gamma2);
set_param(file,'StopTime','60');
set_param('ex5/termoX1','Gain',num2str(param1));
set_param('ex5/termoX2','Gain',num2str(param2));
set_param(file,'FixedStep','0.001');
set_param(file,'StopTime','10');
set_param('ex5/U','Gain','0');
mod=sim(file,'SimulationMode','Normal');

x1=mod.get('x1');
x2=mod.get('x2');
clk=mod.get('clk');

figure 
plot(clk,x1)
xlabel('Time(s)');
ylabel('Angle(rad)');
title('Angle evolution')
figure 
plot(clk,x2)
xlabel('Time(s)');
ylabel('d theta');
title('Angle derivative evolution')

figure 
plot(x1,x2)
xlabel('\beta', 'Interpreter','Latex');
ylabel('$$ {d\over dt }\theta $$','Interpreter','Latex');
title('Phase space');
%% 
% Comparing both we see that there are 2 differences, amplitude and phase
% difference between the curves. They will bot stabilize at the origin
% In the phase space we see that the system oscillates going arround the
% origin, with the amplitude going lower.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ex6
% 
save_system('ex5')
close_system('ex5');

load_system('ex6');
file='ex6';
set_param('ex6/State-Space','A','[0 1;-41.5393 -2.4719]');
set_param('ex6/State-Space','B','[0;2.4719]');
set_param('ex6/State-Space','C','[1 0;0 1]');
set_param('ex6/State-Space','D','[0;0]');
set_param('ex6/State-Space','X0','[0 ;pi/4]');

set_param(file,'FixedStep','0.0001');
set_param(file,'StopTime','10');
mod=sim(file,'SimulationMode','Normal');

yout=mod.get('yout');
clk=mod.get('clk');


figure 
plot(clk,yout.signals.values(:,1));
xlabel('Time(s)');
ylabel('Angle(rad)');
title('Angle evolution')
figure 
plot(clk,yout.signals.values(:,2));
xlabel('Time(s)');
ylabel('d theta');
title('Angle derivative evolution')
figure 
plot(yout.signals.values(:,1),yout.signals.values(:,2));
xlabel('\beta', 'Interpreter','Latex');
ylabel('$$ {d\over dt }\theta $$','Interpreter','Latex');
title('Phase space');
%%
% Whit the linear method (matrix form), this system and the previous are
% similar so, we get the same results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ex7
% 
%% $$ \beta = 1 $$
% 
gamma1=0.1175;
gamma2=0.0445;
g=9.8;
k=3;
beta=[0 1];

param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(beta(1)/gamma2);
paramu=1/gamma2;

param1=num2str(param1);
param2=num2str(param2);
paramu=num2str(paramu);

paramA=strcat('[0 1;',param1,32,param2,']');    %32 is space bar in ASCII
paramB=strcat('[0;',paramu,']');

set_param('ex6/State-Space','A',paramA);
set_param('ex6/State-Space','B',paramB);
set_param('ex6/State-Space','C','[1 0;0 1]');
set_param('ex6/State-Space','D','[0;0]');
set_param('ex6/State-Space','X0','[0 ;pi/4]');

set_param(file,'FixedStep','0.0001');
set_param(file,'StopTime','10');
mod=sim(file,'SimulationMode','Normal');
%get values
yout=mod.get('yout');
y1=yout.signals.values(:,1);
y2=yout.signals.values(:,2);
clk=mod.get('clk');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot
figure 
plot(clk,y1);
xlabel('Time(s)');
ylabel('Angle(rad)');
title('Angle evolution')
figure 
plot(clk,y2);
xlabel('Time(s)');
ylabel('d theta');
title('Angle derivative evolution')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%quiver time 
param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(beta(1)/gamma2);
x=linspace(min(y1),max(y1),20);
y=linspace(min(y2),max(y2),20);
[X,Y]=meshgrid(x,y);
u=Y;
v=param1.*X + param2.*Y;

figure
quiver(X,Y,u,v,0.5)
hold on
plot(y1,y2);
init=linspace(-pi/4,pi/4,10);
for i=1:10
    some=num2str(init(i));
    aux2=strcat('[0 ;',some,']');
    set_param('ex6/State-Space','X0',aux2);
    mod=sim(file,'SimulationMode','Normal');
    yout=mod.get('yout');
    y1=yout.signals.values(:,1);
    y2=yout.signals.values(:,2);
    plot(y1,y2)
end
xlabel('\theta','fontweight','bold','Interpreter','tex');
ylabel('$$ {d\over dt }\theta $$','fontweight','bold','Interpreter','Latex');
title('Espaço de Estados');


%% $$ \beta = 0 $$
% 
gamma1=0.1175;
gamma2=0.0445;
g=9.8;
k=3;
beta=[0 1];

param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(beta(2)/gamma2);
paramu=1/gamma2;

param1=num2str(param1);
param2=num2str(param2);
paramu=num2str(paramu);

paramA=strcat('[0 1;',param1,32,param2,']');    %32 is space bar in ASCII
paramB=strcat('[0;',paramu,']');

set_param('ex6/State-Space','A',paramA);
set_param('ex6/State-Space','B',paramB);
set_param('ex6/State-Space','C','[1 0;0 1]');
set_param('ex6/State-Space','D','[0;0]');
set_param('ex6/State-Space','X0','[0 ;pi/4]');

set_param(file,'FixedStep','0.0001');
set_param(file,'StopTime','10');
mod=sim(file,'SimulationMode','Normal');
%get values
yout=mod.get('yout');
y1=yout.signals.values(:,1);
y2=yout.signals.values(:,2);
clk=mod.get('clk');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot
figure 
plot(clk,y1);
figure 
plot(clk,y2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%quiver time and convert from rad to degrees for better understanding
param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(beta(2)/gamma2);
x=linspace(-0.04,0.05,20);
y=linspace(-0.9,0.9,20);
[X,Y]=meshgrid(x,y);
u=Y;
v=param1.*X + param2.*Y;

figure
quiver(X,Y,u,v,0.5)
hold on
plot(y1,y2);
init=linspace(-pi/4,pi/4,10);
for i=1:10
    some=num2str(init(i));
    aux2=strcat('[0 ;',some,']');
    set_param('ex6/State-Space','X0',aux2);
    mod=sim(file,'SimulationMode','Normal');
    yout=mod.get('yout');
    y1=yout.signals.values(:,1);
    y2=yout.signals.values(:,2);
    plot(y1,y2)
end
%% 
% We begin by analyzing the case where $$ \beta = 0 $$. In this case, there is no
% friction, and as such, the regime, oscillates indefinitely, since they are not
% forces the movement of the metronome. For the second
% case, where Beta = 1, the coefficient of friction is quite high, and as
% such a system, it quickly tends to lose its movement, and tends to
% equilibrium position. In the states plan, we observe exactly this
% effect, since for Beta = 0, we observed a circumnar trajectory in
% around the origin, and that in the case where Beta = 1, we observe a curve that
% the origin, without oscillating around the
% source.
% For the case where the friction is null, since force is not realized by
% part of the spring, the system oscillates for unlimited time around the origin,
% describing an ellipse in the plane of states. For the other case, where the
% friction is not zero, the system does not loop, and tends to the origin,
% which is its point of equilibrium.




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EigenVectors 
%
[V,D]=eig([0 1;(-(k/gamma2)+(gamma1/gamma2)*g) -(1/gamma2)]);
V=real(V);
display('For beta=1 we get the following eigenvectors:')
display(V)
display('For beta=1 we get the following eigenvalues:')
display(D)
[V,D]=eig([0 1;(-(k/gamma2)+(gamma1/gamma2)*g) -(0/gamma2)]);
V=real(V);
display('For beta=0 we get the following eigenvectors:')
display(V)
display('For beta=0 we get the following eigenvalues:')
display(D)
[V,D]=eig([0 1;(-(k/gamma2)+(gamma1/gamma2)*g) -(0.1/gamma2)]);
V=real(V);
display('For beta=0.1 we get the following eigenvectors:')
display(V)
display('For beta=0.1 we get the following eigenvalues:')
display(D)

%% Ex8
% 

param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(1/gamma2);
paramu=1/gamma2;

A=[0 1;param1 param2];
[V,D]=eig(A);

param1=num2str(param1);
param2=num2str(param2);
paramu=num2str(paramu);

paramA=strcat('[0 1;',param1,32,param2,']');    %32 is space bar in ASCII
paramB=strcat('[0;',paramu,']');

set_param('ex6/State-Space','A',paramA);
set_param('ex6/State-Space','B',paramB);
set_param('ex6/State-Space','C','[1 0;0 1]');
set_param('ex6/State-Space','D','[0;0]');

for i=1:2
    some=num2str(V(1,i));
    aux2=num2str(V(2,i));
    init=strcat('[',some,';',aux2,']');
    set_param('ex6/State-Space','X0',init);
    mod=sim(file,'SimulationMode','Normal');
    yout=mod.get('yout');
    y1=yout.signals.values(:,1);
    y2=yout.signals.values(:,2);
    figure
    plot(y1,y2);
    xlabel('\theta','fontweight','bold','Interpreter','tex');
    ylabel('$$ {d\over dt }\theta $$','fontweight','bold','Interpreter','Latex');
    title('Linear Phase space for $$ \beta = 1 $$','Interpreter','Latex');
end
%%
% Using the data from the previous question (the eigenvalues and vectors). 
% Beta = 1 was used. State trajectories as can be seen, are a straight line,
% as desired.




%% Ex9
% 

m_t=linspace(0,0.2,100);
l_t=linspace(0.05,0.25,100);
grid=NaN(100,100);
%Variables
beta = 0.001;
k=0.35;
M=0.1;
L=0.25;

for i=1:100
    for j=1:100
        m=m_t(i);
        l=l_t(j);
        gamma1 = l*m + (L/2)*M;
        gamma2 = (1/3)*M*(L.^2) + m *l.^2;
        wn = sqrt((k-gamma1*9.8)/gamma2);
        dt = ((beta)/gamma2)*(1/(2*wn));
        
        wa=wn*sqrt(1-dt^2);
        fa=wa/(2*pi);
        BPM=120*fa;
        if (real(BPM)~=0)
            grid(i,j)=real(BPM);
        end
    end
end

figure
mesh(m_t,l_t,grid);
xlabel('Mass [m]');
ylabel('Length [l]');
zlabel('BPM');
colormap
colorbar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BPM given
bpm1=130; %allegro
bpm2=64;   %lento
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Check if mass values correspond to what we want
for i=1:100
    some=0;
    aux2=0;
    if(max(grid(i,:))>=bpm1)
        some=1;
    end
    if(min(grid(i,:))<=bpm2)
        aux2=1;
    end
    if(aux2==1 &&some==1)
        THIS_M=m_t(i);
        break;
    end
end
%Get length values
for i=1:100
        l=l_t(i);
        gamma1 = l*m + (L/2)*M;
        gamma2 = (1/3)*M*(L.^2) + m *l.^2;
       
        wn = sqrt((k-gamma1*9.8)/gamma2);
        dt = ((beta)/gamma2)*(1/(2*wn));
        wa=wn*sqrt(1-dt^2);
        fa=wa/(2*pi);
        BPM=120*fa;
        if (real(BPM)~=0)
            vec_bpm(i)=real(BPM);
        end
end



err_a=abs(vec_bpm-bpm1);
err_l=abs(vec_bpm-bpm2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%these are the values we want!!!!!!!!!!
[min_l,index_l]=min(err_l);
[min_a,index_a]=min(err_a);
THIS_M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sim lento
xinit='[pi/4;0]';
l=l_t(index_l);
gamma1 = l*m + (L/2)*M;
gamma2 = (1/3)*M*(L.^2) + m *l.^2;

param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(0.001/gamma2);
paramu=1/gamma2;

param1=num2str(param1);
param2=num2str(param2);
paramu=num2str(paramu);

paramA=strcat('[0 1;',param1,32,param2,']');    %32 is space bar in ASCII
paramB=strcat('[0;',paramu,']');

set_param('ex6/State-Space','A',paramA);
set_param('ex6/State-Space','B',paramB);
set_param('ex6/State-Space','C','[1 0;0 1]');
set_param('ex6/State-Space','D','[0;0]');
set_param(file,'FixedStep','0.001');
set_param(file,'StopTime','60');
set_param('ex6/State-Space','X0',xinit);
mod=sim(file,'SimulationMode','Normal');
yout=mod.get('yout');
y1=yout.signals.values(:,1);
clk=mod.get('clk');

wn = sqrt((k-gamma1*9.8)/gamma2);
dt = ((beta)/gamma2)*(1/(2*wn));
time=linspace(0,60,120);
env=(pi/4)*exp(-(dt*wn*time));

[pks,locs] = findpeaks(y1);
for i=1:(length(pks)-1)
   some(i)=clk(locs(i+1))-clk(locs(i));
end

med=mean(some);
aux_bpm=(1/med)*120;
par1=num2str(bpm2);
par2=num2str(aux_bpm);
phrase=strcat('Wanted BPM:',par1,', Real BPM:',par2);

%plot env and angle
figure
plot(clk, y1);
hold on
xlabel('Time(s)');
ylabel('Angle(rad)');
plot(time,env,'m--');
plot(time,-env,'y--','color','b');
legend('Angle evoluton','Upper surrounding','Lower surrounding');
title(phrase);
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sim alegro
xinit='[pi/4;0]';
l=l_t(index_a);
gamma1 = l*m + (L/2)*M;
gamma2 = (1/3)*M*(L.^2) + m *l.^2;

param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(0.001/gamma2);
paramu=1/gamma2;

param1=num2str(param1);
param2=num2str(param2);
paramu=num2str(paramu);

paramA=strcat('[0 1;',param1,32,param2,']');    %32 is space bar in ASCII
paramB=strcat('[0;',paramu,']');

set_param('ex6/State-Space','A',paramA);
set_param('ex6/State-Space','B',paramB);
set_param('ex6/State-Space','C','[1 0;0 1]');
set_param('ex6/State-Space','D','[0;0]');
set_param(file,'StopTime','60');
set_param('ex6/State-Space','X0',xinit);
mod=sim(file,'SimulationMode','Normal');
yout=mod.get('yout');
y1=yout.signals.values(:,1);
clk=mod.get('clk');

wn = sqrt((k-gamma1*9.8)/gamma2);
dt = ((beta)/gamma2)*(1/(2*wn));
time=linspace(0,60,120);
env=(pi/4)*exp(-(dt*wn*time));

[pks,locs] = findpeaks(y1);
for i=1:(length(pks)-1)
   some(i)=clk(locs(i+1))-clk(locs(i));
end

med=mean(some);
aux_bpm=(1/med)*120;
par1=num2str(bpm1);
par2=num2str(aux_bpm);
phrase=strcat('Allegro -> Wanted BPM:',par1,', Real BPM:',par2);

%plot env and angle
figure
plot(clk, y1);
hold on
xlabel('Time(s)');
ylabel('Angle(rad)');
plot(time,env,'m--');
plot(time,-env,'y--','color','b');
legend('Angle evoluton','Upper surrounding','Lower surrounding');
title(phrase);

%% 
% A routine was created that was able to choose a mass for which
% the two beats per minute were possible.
% The number of oscillations corresponds to the desired number, apart from small
% variations.






%% Ex10
% 
save_system('ex6')
close_system('ex6');

load_system('ex5');
file='ex5';
bpm_test=[bpm1 bpm2];
l_test=[l_t(index_a) l_t(index_l)];
beta = 0.001;
k=0.35;
M=0.1;
L=0.25;

for i=1:2
    l=l_test(i);
    gamma1 = l*m + (L/2)*M;
    gamma2 = (1/3)*M*(L.^2) + m *l.^2;
    param1=-(k/gamma2) + (gamma1/gamma2)*g;
    param2=-(beta/gamma2);
    set_param(file,'StopTime','60');
    set_param('ex5/termoX1','Gain',num2str(param1));
    set_param('ex5/termoX2','Gain',num2str(param2));
    set_param('ex5/Integrator','InitialCondition','0');
    set_param('ex5/Integrator1','InitialCondition','pi/4');
    mod=sim(file,'SimulationMode','Normal');
    y1=mod.get('x1');
    clk=mod.get('clk');
    figure;
    plot(clk, y1);
    [pks ,locs] = findpeaks(y1);
    for j=1:(length(pks)-1)
        some(j)=clk(locs(j+1))-clk(locs(j));
    end
    aux_med=mean(some);
    aux_bpm(i)=(1/aux_med)*120;
    par1=num2str(bpm_test(i));
    par2=num2str(aux_bpm(i));
    phrase=strcat('Wanted BPM:',par1,', Real BPM:',par2);
    title(phrase);
    xlabel('Time (s)');
    ylabel('Angle');
end
    %L Dimensioning
m1=linspace(-0.02,0,100);
m2=linspace(0,0.02,100);
ma=[m1,m2(2:end)];
m_l=linspace(-0.08,0,100);
m_l2=linspace(0,0.08,100);
m_lf=[m_l,m_l2(2:end)];

lent=m_lf+l_test(2);
al=ma+l_test(1);
dif_L=5;
dif_A=5;
%Ritmo lento com massa Compensada
for i=1:length(ma)
    l=lent(i);
    gamma1 = l*m + (L/2)*M;
    gamma2 = (1/3)*M*(L.^2) + m *l.^2;
    param1=-(k/gamma2) + (gamma1/gamma2)*g;
    param2=-(beta/gamma2);
    set_param(file,'StopTime','60');
    set_param('ex5/termoX1','Gain',num2str(param1));
    set_param('ex5/termoX2','Gain',num2str(param2));
    set_param('ex5/Integrator','InitialCondition','0');
    set_param('ex5/Integrator1','InitialCondition','pi/4');
    mod=sim(file,'SimulationMode','Normal');
    y1=mod.get('x1');
    clk=mod.get('clk');
    [pks ,locs] = findpeaks(y1);
    some=zeros(1,length(peaks)-1);
    for j=1:(length(pks)-1)
        some(j)=clk(locs(j+1))-clk(locs(j));
    end
    aux_med=mean(some);
    aux_bpm=(1/aux_med)*120;
    par1=num2str(bpm_test(2));
    par2=num2str(aux_bpm);
    if(abs(aux_bpm-bpm_test(2))<=dif_L)
        figure;
        plot(clk, y1);
        bpm_lento=aux_bpm;
        phrase=strcat('Wanted BPM:',par1,', Real BPM:',par2);
        title(phrase);  
        break
    end
end


%Ritmo rápido com massa compensada
for i=1:length(ma)
    l=al(i);
    gamma1 = l*m + (L/2)*M;
    gamma2 = (1/3)*M*(L.^2) + m *l.^2;
    param1=-(k/gamma2) + (gamma1/gamma2)*g;
    param2=-(beta/gamma2);
    set_param(file,'StopTime','60');
    set_param('ex5/termoX1','Gain',num2str(param1));
    set_param('ex5/termoX2','Gain',num2str(param2));
    set_param('ex5/Integrator','InitialCondition','0');
    set_param('ex5/Integrator1','InitialCondition','pi/4');
    mod=sim(file,'SimulationMode','Normal');
    
    y1=mod.get('x1');
    clk=mod.get('clk');
    [pks ,locs] = findpeaks(y1);
    some1=zeros(1,length(peaks)-1);
    for j=1:(length(pks)-1)
        some1(j)=clk(locs(j+1))-clk(locs(j));
    end
    aux_med=mean(some1);
    aux_bpm=(1/aux_med)*120;
    if(abs(aux_med-bpm_test(1))<=dif_A)
        figure
        plot(clk, y1);
        par1=num2str(bpm_test(1));
        par2=num2str(aux_bpm);
        phrase=strcat('Wanted BPM:',par1,', Real BPM:',par2);
        title(phrase)  
        break
    end
end

%%
% The previously obtained mass and length values were inserted
% in a new scheme, which simulated the non-linearized system. The results
% show a small difference between the BPM values that would be
% wait and those that were actually obtained.
%
% As a strategy of mass optimization, we have created a vector of lengths
% possible for each of the lengths. In the end we get the one who led to
% the least difference desired.

%% Ex11
% 
save_system('ex5')
close_system('ex5');

load_system('ex11');
file='ex11';

l=l_test(1);
k=0.35;
M=0.1;
L=0.25;
T_teste=[0 0.1];
gamma1 = l*m + (L/2)*M;
gamma2 = (1/3)*M*(L.^2) + m *l.^2;
param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(beta/gamma2);
set_param(file,'StopTime','60');
set_param('ex11/termoX1','Gain',num2str(param1));
set_param('ex11/termoX2','Gain',num2str(param2));
set_param('ex11/Integrator','InitialCondition','0');
set_param('ex11/Integrator1','InitialCondition','pi/4');


%Without binary
	T=T_teste(1);
    set_param('ex11/Gain3','Gain',num2str(T));
    set_param('ex11/Gain2','Gain',num2str(T));
    mod=sim(file,'SimulationMode','Normal');
    y1w=mod.get('x1');
    clkw=mod.get('clk');
    binw=mod.get('bin');
    
    [pks,locs] = findpeaks(y1);
    for k=1:(length(pks)-1)
        som3(k)=clk(locs(k+1))-clk(locs(k));
    end
    aux_med=mean(som3);
    aux_bpm=(1/aux_med)*120;
    par1=num2str(bpm_test(1));
    par2=num2str(aux_bpm);
    phrase=strcat('Wanted BPM:',par1,', Real BPM:',par2);
    
    
%With binary
    T=T_teste(2);
    set_param('ex11/Gain3','Gain',num2str(T));
    set_param('ex11/Gain2','Gain',num2str(T));
    mod=sim(file,'SimulationMode','Normal');
    y1b=mod.get('x1');
    clkb=mod.get('clk');
    binb=mod.get('bin');     
   

figure
plot(clkw,y1w,clkb,y1b)
legend('Without binary', 'With');
title(phrase)
figure
plot(clkw,binw,clkb,binb)
ylabel('Applied binary');
xlabel('Time');
title('Time evolution of binary');

%%
% The results were not as expected

%% Ex12

gamma1 = l*m + (L/2)*M;
gamma2 = (1/3)*M*(L.^2) + m *l.^2;
param1=-(k/gamma2) + (gamma1/gamma2)*g;
param2=-(beta/gamma2);

figure
for i=1:2
    l=l_test(i);
    gamma2 = (1/3)*M*(L.^2) + m *l.^2;
    numerador=1/gamma2;
    denominador=[1 beta/gamma2 (k-g*(m*l+M*L/2))/gamma2];
    
    bode(tf(numerador, denominador));
    hold on;
    grid;
    title(sprintf('Diagrama de Bode para l=%.4f',l));
end

%%
% As one would expect, different values of l do not cause a change
% of the shape of the Bode diagram. They cause a displacement of the curve, either
% in terms of both amplitude and frequency.
% We conclude that this displacement corresponds to the change of torque
% which is necessary to ensure that the oscillatory regime is maintained by
% indeterminate time (since friction disagreements occur).
% Thus, the larger the value of l, the smaller the torque that is needed
% to apply.


end