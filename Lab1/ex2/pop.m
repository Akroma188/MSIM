%% Study of populations  
% Here we evaluate a a physhical model of a car going forward taking in
% account different values of the contant of time which is given by the friction constant
% divided by mass

%% Load model
function pop

close all;

%Load file in system memory
file='popul';
load_system(file);
%% Variable Definition
StopTime='20';
N1_init='2';     %Prey - Bottom block of model
N2_init='2';     %Predadator - Top block of model
lambda1=2;
lambda2=1;
gamma1=1;
gamma2=2;
alpha1=1;
alpha2=1;

%Define strings to input into model
delta1=lambda1-gamma1;
delta2=lambda2-gamma2;

term1=strcat(delta1,'u(1)');
alpha1=num2str(alpha1);
aux=strcat('-',alpha1);
term2=strcat(aux,'*u(1)*u(2)');
Prey_fcn=strcat(term1,term2);
%if(alpha1<0)
%    alpha1=num2str(alpha1);
%    aux=strcat('+',alpha1);
%    term2=strcat(aux,'*u(1)*u(2)');
%else
%    alpha1=num2str(alpha1);
%    aux=strcat('-',alpha1);
%    term2=strcat(aux,'*u(1)*u(2)');
%end
Prey_fcn=strcat(term1,term2);

%% 2.1
% Simulations with $$\alpha_1 = \alpha_2 $$ and for various values of
% $$\sigma_1 $$ and $$\sigma_2 $$

set_param(file,'StopTime',StopTime);
set_param('popul/Integrator1','InitialCondition',N1_init);
set_param('popul/Integrator','InitialCondition',N2_init);

set_param('popul/Fcn1','Expr','(1-2+2)*u(1)-1*u(1)*u(2)');
set_param('popul/Fcn2','Expr','(1-2)*u(2) +1*u(2)*u(1)');

mod=sim(file,'SimulationMode','Normal');

clk=mod.get('clock');
N2=mod.get('data');
N1=mod.get('data1');




end