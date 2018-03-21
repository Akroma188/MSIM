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
StopTime='4';
N1_init='2';     
N2_init='2';     
lambda1=2;
lambda2=1;
gamma1=1;
gamma2=2;
alpha1=1;
alpha2=1;

%Define strings to input into model
delta1=lambda1-gamma1;
delta2=lambda2-gamma2;

delta1=num2str(delta1);
term1_N1=strcat(delta1,'*u(1)');
alpha1=num2str(alpha1);
aux=strcat('-',alpha1);
term2_N1=strcat(aux,'*u(1)*u(2)');
prey_fcn=strcat(term1_N1,term2_N1);

delta2=num2str(delta2);
term1_N2=strcat(delta2,'*u(2)');
alpha2=num2str(alpha2);
aux2=strcat('+',alpha2);
term2_N2=strcat(aux2,'*u(1)*u(2)');
pred_fcn=strcat(term1_N2,term2_N2);

%% 2.1
% Simulations with $$\alpha_1 = \alpha_2 $$ and for various values of
% $$\sigma_1 $$ and $$\sigma_2 $$
for i=1:3
    delta1=[-1 0 1];
    delta2=[-1 0 1];

    for k=1:3
        delt1=num2str(delta1(i));
        term1_N1=strcat(delt1,'*u(1)');
        alpha1=num2str(alpha1);
        aux=strcat('-',alpha1);
        term2_N1=strcat(aux,'*u(1)*u(2)');
        prey_fcn=strcat(term1_N1,term2_N1);

        delt2=num2str(delta2(k));
        term1_N2=strcat(delt2,'*u(2)');
        alpha2=num2str(alpha2);
        aux2=strcat('+',alpha2);
        term2_N2=strcat(aux2,'*u(1)*u(2)');
        pred_fcn=strcat(term1_N2,term2_N2);

        set_param(file,'StopTime',StopTime);
        set_param('popul/Integrator1','InitialCondition',N1_init);
        set_param('popul/Integrator','InitialCondition',N2_init);

        set_param('popul/Fcn1','Expr',prey_fcn);
        set_param('popul/Fcn2','Expr',pred_fcn);
        mod=sim(file,'SimulationMode','Normal');

        clk=mod.get('clock');
        N2=mod.get('dataN2');
        N1=mod.get('dataN1');
        figure
        h=plot(clk,N2,clk,N1);        
    end
end
%% 2.4


end