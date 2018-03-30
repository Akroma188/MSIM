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
alpha1=1;
alpha2=1;
legenda1='Predadator (N2)';
legenda2='Prey (N1)';
xaxe='Time (s)';
yaxe='[Predadator (N2),Prey (N1)]';

%Define strings to input into model


%% 2.1
% Simulations with $$\alpha_1 = \alpha_2 $$ and for various values of
% $$\sigma_1 $$ and $$\sigma_2 $$

%% 2.2
% With $$\alpha_1 = \alpha_2 = 1 $$ we evaluate the different types of evolution
% considering various values for $$\delta_1 $$ and $$\delta_2 $$

mod=sim_plot(alpha1,alpha2,'10','0.0001',N1_init,N2_init,file,-1,-1);
tittle='Interaction between the two populations with \delta_1 = -1 and \delta_2 = -1';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = -1 $$ and $$\delta_2 = -1 $$  osccilates
mod=sim_plot(alpha1,alpha2,StopTime,'0.0001',N1_init,N2_init,file,-1,0);
tittle='Interaction between the two populations with \delta_1 = -1 and \delta_2 = 0';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = -1 $$ and $$\delta_2 = 0 $$ equilibrium
mod=sim_plot(alpha1,alpha2,'2','0.0001',N1_init,N2_init,file,-1,1);
tittle='Interaction between the two populations with \delta_1 = -1 and \delta_2 = 1';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = -1 $$ and $$\delta_2 = 1 $$

mod=sim_plot(alpha1,alpha2,'10','0.0001',N1_init,N2_init,file,0,-1);
tittle='Interaction between the two populations with \delta_1 = 0 and \delta_2 = -1';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = 0 $$ and $$\delta_2 = -1 $$  osccilates
mod=sim_plot(alpha1,alpha2,StopTime,'0.0001',N1_init,N2_init,file,0,0);
tittle='Interaction between the two populations with \delta_1 = 0 and \delta_2 = 0';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = 0 $$ and $$\delta_2 = 0 $$ equilibrium
mod=sim_plot(alpha1,alpha2,'2','0.0001',N1_init,N2_init,file,0,1);
tittle='Interaction between the two populations with \delta_1 = 0 and \delta_2 = 1';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = 0 $$ and $$\delta_2 = 1 $$

mod=sim_plot(alpha1,alpha2,'10','0.0001',N1_init,N2_init,file,1,-1);
tittle='Interaction between the two populations with \delta_1 = 1 and \delta_2 = -1';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = 1 $$ and $$\delta_2 = -1 $$  osccilates
mod=sim_plot(alpha1,alpha2,StopTime,'0.0001',N1_init,N2_init,file,1,0);
tittle='Interaction between the two populations with \delta_1 = 1 and \delta_2 = 0';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = 1 $$ and $$\delta_2 = 0 $$ equilibrium
mod=sim_plot(alpha1,alpha2,'2','0.0001',N1_init,N2_init,file,1,1);
tittle='Interaction between the two populations with \delta_1 = 1 and \delta_2 = 1';
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
%%
% $$\delta_1 = 1 $$ and $$\delta_2 = 1 $$


%% 2.3
% To confirm equilibrium we need to assure that $$N_1 =
% -\frac{\delta_2}{\alpha_2} $$ and $$\frac{\delta_1}{\alpha_1} $$, so
% ploting the case where the populations osccilate i.e for every $$\delta_2 < 0 $$

mod=sim_plot(alpha1,alpha2,'10','0.0001','1','3',file,3,-1);
plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);
ylim([0 4]);

%% 2.4 a)
% Here we plot presas.mat togeter with our best guess of values for the
% predadotr evolution
mod=sim_plot(1.4,0.71,'20','0.0001','4','1.47',file,3.1,-1.5);
%plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe);

prey=load('presas.mat');
h=plot(prey.tr,prey.yr,'linewidth',2);
N1=mod.get('dataN1');
clk=mod.get('clock');
hold on;
h1=plot(clk,N1,'linewidth',2);
grid on;
grid minor;
xlabel('Time (s)');
ylabel('N1 Population');
title('Approximation by visual guidance','Interpreter','tex');
legend([h,h1],'N1 given by file','N1 by visual approximation');
hold off
%%
% We can see by the maximum values of presas.mat that our values for the
% predadator function are a good estimate of the real ones

%% 2.4 b)
%

alpha2_t=linspace(0.68,0.72,20);
N2_t=linspace(1.58,1.62,20);
[X,Y]=meshgrid(alpha2_t,N2_t);
Z = error_matrix(alpha2_t,N2_t);

h=figure;
create_mesh(h,X,Y,Z);    %made by righ clicking the figure (../get code) because i rotated it to get a better view
%Get the min error value and index
I=find(Z==min(min(Z)));
N2_init=num2str(Y(I));
alpha_2=num2str(X(I));

%Plot bot prey evolution
mod=sim_plot(1.4,alpha_2,'20','0.0001','4',N2_init,file,3.1,-1.5);
N1=mod.get('dataN1');
clk=mod.get('clock');
figure
h=plot(prey.tr,prey.yr,'linewidth',2);
hold on;
h1=plot(clk,N1,'linewidth',2);
xlabel('Time (s)');
ylabel('N1 Population');
title('Approximation by brute force','Interpreter','tex');
legend([h,h1],'N1 given by file','N1 by approximation');
grid on;
grid minor;
hold off;
N2_init=num2str(Y(I));
alpha_2=num2str(X(I));
X = sprintf('By this iteration we get N(0)=%s and alpha_2 =%s as our best values for aproximation',N2_init,alpha_2);
disp(X)

%%
% Using brute force iterations we see that we can get a really good
% approximation. The time it takes
%% 2.4 d)
% Optimization usig fminunc
fh=@algorithm1;
x0=[0.60;1.50];
[x, fval]=fminunc(fh,x0)
mod=sim_plot(1.4,alpha_2,'20','0.0001','4',N2_init,file,3.1,-1.5);

x
fval
%%
% Comentar aqui BLA BLA BLA
       



%% 2.4 c)
%



end



function mod = sim_plot(alpha1,alpha2,StopTime,Step,N1_init,N2_init,file,delta1,delta2)

        delt1=num2str(delta1);
        term1_N1=strcat(delt1,'*u(1)');
        alpha1=num2str(alpha1);
        aux=strcat('-',alpha1);
        term2_N1=strcat(aux,'*u(1)*u(2)');
        prey_fcn=strcat(term1_N1,term2_N1);

        delt2=num2str(delta2);
        term1_N2=strcat(delt2,'*u(2)');
        alpha2=num2str(alpha2);
        aux2=strcat('+',alpha2);
        term2_N2=strcat(aux2,'*u(1)*u(2)');
        pred_fcn=strcat(term1_N2,term2_N2);
        
        N1_init=num2str(N1_init);
        N2_init=num2str(N2_init);
        
        set_param(file,'StopTime',StopTime);
        set_param('popul/Integrator','InitialCondition',N1_init);
        set_param('popul/Integrator1','InitialCondition',N2_init);

        set_param('popul/Fcn1','Expr',prey_fcn);
        set_param('popul/Fcn2','Expr',pred_fcn);
        set_param(file,'FixedStep',Step);
        mod=sim(file,'SimulationMode','Normal');
end

function plot_pop(mod,tittle,legenda1,legenda2,xaxe,yaxe)

        clk=mod.get('clock');
        N2=mod.get('dataN2');
        N1=mod.get('dataN1');
        V=figure;
        h=plot(clk,N2,'linewidth',2); 
        hold on;
        h1=plot(clk,N1,'linewidth',2);
        xlabel(xaxe);
        ylabel(yaxe);
        title(tittle,'Interpreter','tex');
        legend([h,h1],legenda1,legenda2);
        grid on;
        grid minor;
        hold off
end

function Z = error_matrix(alpha2_t,N2_t)
[X,Y]=meshgrid(alpha2_t,N2_t);
sz=size(X);
Z=zeros(sz(1),sz(2));
h = waitbar(0,'Please wait...');
for i=1:sz(1)
    msg=num2str(i);
    for j=1:sz(2)
        %messagge box
        msg1=num2str(j);
        str=strcat(msg,'/');
        str=strcat(str,msg1);
        waitbar(i/sz(1),h,str);
        %sim and ge error
        x(2)=Y(i,j);
        x(1)=X(i,j);
        Z(i,j)=algorithm1(x);
    end
end
close(h)

end

function total_error= algorithm1(x)
    prey=load('presas.mat');
    mod=sim_plot(1.4,x(1),'20','0.1','4',x(2),'popul',3.1,-1.5);
    N1=mod.get('dataN1');
    total_error=0;
    for i=1:size(prey.yr)     
        erro= abs(prey.yr(i)-N1(i));
        total_error=total_error+erro;
    end
    
end


function create_mesh(Parent1, xdata1, ydata1, zdata1)
%CREATEAXES(PARENT1, XDATA1, YDATA1, ZDATA1)
%  PARENT1:  axes parent
%  XDATA1:  surface xdata
%  YDATA1:  surface ydata
%  ZDATA1:  surface zdata

%  Auto-generated by MATLAB on 29-Mar-2018 16:10:25

% Create axes
axes1 = axes('Parent',Parent1);
hold(axes1,'on');

% Create mesh
mesh(xdata1,ydata1,zdata1,'Parent',axes1);

% Create xlabel
xlabel('alpha1');

% Create zlabel
zlabel('Total Error(%)');

% Create title
title('Error calculation');

% Create ylabel
ylabel('N1(0)');

colorbar;

view(axes1,[130.5 8.4000000000001]);
grid(axes1,'on');
end




