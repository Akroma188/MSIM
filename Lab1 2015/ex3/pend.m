function pend

close all;
%Load file in system memory
file='pendulo';
load_system(file);

%% 3.1 Animation
% Remove commented section (select all the lines and press CTRL+T) (To comment everything again sellect everything and press CTRL)

theta1i=pi/2;
theta2i=pi/2;
p1i=0;
p2i=0;

set_simulation_parameters(1,0.5,9.8,theta1i,theta2i,p1i,p2i,'10','0.01',file,'0.5','0','1','0');
mod=sim(file,'SimulationMode','Normal');
theta1=mod.get('theta1');
theta2=mod.get('theta2');
p1=mod.get('p1');
p2=mod.get('p2');
x1=mod.get('x1');
y1=mod.get('y1');
x2=mod.get('x2');
y2=mod.get('y2');
clk=mod.get('clock');
sz=size(clk);

figure
axis([-1.2 1.2 -1.2 1.2])

for k=1 : sz(1)
    plot(0,0,'x')
    xlim([-1.1 1.1])
    ylim([-1.1 1.1])
    hold on
    plot([0 x1(k)],[0 y1(k)],'linewidth',2)
    plot(x1(k),y1(k),'o','MarkerFaceColor', 'b');
    plot(x2(k),y2(k),'o','MarkerFaceColor', 'r');
    plot([x1(k) x2(k)],[y1(k) y2(k)],'linewidth',2)  
    pause(0.01)
    clf
end

%% 3.2 Lissajous Curve
%   For small initial $$\theta_1 $$ and $$\theta_2 $$ conditions
theta1i=pi/20;
theta2i=pi/20;
p1i=0;
p2i=0;

set_simulation_parameters(1,0.5,9.8,theta1i,theta2i,p1i,p2i,'10','0.01',file,'0.5','0','1','0');
mod=sim(file,'SimulationMode','Normal');
theta1=mod.get('theta1');
theta2=mod.get('theta2');
figure
plot(theta1,theta2,'k');
xlabel('\theta_1 (rad)','fontweight','bold','Interpreter','tex');
ylabel('\theta_2 (rad)','fontweight','bold','Interpreter','tex');
title('Lissajous Curve','Interpreter','tex');

end

function set_simulation_parameters(m,l,g,th1_init,th2_init,p1_init,p2_init,StopTime,Step,file,x1_init,y1_init,x2_init,y2_init)
    set_param(file,'StopTime',StopTime);
    set_param(file,'FixedStep',Step);
    
    th1_init=num2str(th1_init);
    th2_init=num2str(th2_init);
    p1_init=num2str(p1_init);
    p2_init=num2str(p2_init);

    d_theta1=fcn_theta1(m,l)
    d_theta2= fcn_theta2(m,l)
    d_p1=fcn_p1(m,l,g)
    d_p2=fcn_p2(m,l,g)
    str5=fcn_x1(l)
    str6=fc_y1(l)
    str7=fcn_x2(l)
    str8=fcn_y2(l)
    
    %define functions
    set_param('pendulo/d_theta1','Expr',d_theta1);
    set_param('pendulo/d_theta2','Expr',d_theta2);
    set_param('pendulo/d_p1','Expr',d_p1);
    set_param('pendulo/d_p2','Expr',d_p2);
    set_param('pendulo/x1','Expr',str5);
    set_param('pendulo/y1','Expr',str6);
    set_param('pendulo/x2','Expr',str7);
    set_param('pendulo/y2','Expr',str8);
    set_param('pendulo/Integrator','InitialCondition',th1_init);
    set_param('pendulo/Integrator1','InitialCondition',th2_init);
    set_param('pendulo/Integrator2','InitialCondition',p1_init);
    set_param('pendulo/Integrator3','InitialCondition',p2_init);
    set_param('pendulo/Integrator4','InitialCondition',x1_init);
    set_param('pendulo/Integrator5','InitialCondition',y1_init);
    set_param('pendulo/Integrator6','InitialCondition',x2_init);
    set_param('pendulo/Integrator7','InitialCondition',y2_init);
    

   
end

%% Set function into strings
%
function str=fcn_theta1(m,l)
gain=6/(m*l^2);
gain=num2str(gain);
str=strcat(gain,'*(2*u(3)-3*cos(u(1)-u(2))*u(4))/(16-9*cos(u(1)-u(2))^2)');
end

function str1= fcn_theta2(m,l)
gain=6/(m*l^2);
Gain=num2str(gain);
str1=strcat(Gain,'*(8*u(4)-3*cos(u(1)-u(2))*u(3))/(16-9*cos(u(1)-u(2))^2)');
end

function str2 = fcn_p1(m,l,g)
gain=-0.5*m*l^2;
Gain=num2str(gain);

gain2=3*g/l;
Gain2=num2str(gain2);

term1=strcat(Gain,'*(u(1)*u(2)*sin(u(3)-u(4))+');
term2=strcat(Gain2,'*sin(u(3)))');
str2=strcat(term1,term2);
end

function str3 = fcn_p2(m,l,g)
gain=-0.5*m*l^2;
Gain=num2str(gain);

gain2=g/l;
Gain2=num2str(gain2);

term1=strcat(Gain,'*(-u(1)*u(2)*sin(u(3)-u(4))+');
term2=strcat(Gain2,'*sin(u(4)))');
str3=strcat(term1,term2);
end

function str5=fcn_x1(l)
l=num2str(l);
str5=strcat(l,'*u(2)*cos(u(1))');           
end
function str6=fc_y1(l)
l=num2str(l);
str6=strcat(l,'*u(2)*sin(u(1))');
end
function str7=fcn_x2(l)
%l*(sin(u(1))+0.5*sin(u(2)))
l=num2str(l);
str7=strcat(l,'*u(3)*cos(u(1))+');          

aux=strcat(l,'*u(4)*cos(u(2))');
str7=strcat(str7,aux);
end
function str8=fcn_y2(l)
%-l*(cos(u(1))+0.5*cos(u(2)))
l=num2str(l);
str8=strcat(l,'*u(3)*sin(u(1))+');       

aux=strcat(l,'*u(4)*sin(u(2))');
str8=strcat(str8,aux);
end