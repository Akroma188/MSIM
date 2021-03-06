function pend

close all;
%Load file in system memory
file='pendulo';
load_system(file);

%% 3.1 Animation
% Remove commented section (select all the lines and press CTRL+T) (To comment everything again sellect everything and press CTRL)



% 
% set_simulation_parameters(1,0.5,9.8,theta1i,theta2i,p1i,p2i,'10','0.01',file,'0.5','0','1','0');
% mod=sim(file,'SimulationMode','Normal');
% theta1=mod.get('theta1');
% theta2=mod.get('theta2');
% dtheta1=mod.get('dtheta1');
% dtheta2=mod.get('dtheta2');
% p1=mod.get('p1');
% p2=mod.get('p2');
% x1=mod.get('x1');
% y1=mod.get('y1');
% x2=mod.get('x2');
% y2=mod.get('y2');
% clk=mod.get('clock');
% sz=size(clk);
% h=figure
% axis([-1.2 1.2 -1.2 1.2])
% 
% for k=1 : sz(1)
%     plot(0,0,'x')
%     xlim([-1.1 1.1])
%     ylim([-1.1 1.1])
%     hold on
%     plot([0 x1(k)],[0 y1(k)],'linewidth',2)
%     plot(x1(k),y1(k),'o','MarkerFaceColor', 'b');
%     plot(x2(k),y2(k),'o','MarkerFaceColor', 'r');
%     plot([x1(k) x2(k)],[y1(k) y2(k)],'linewidth',2)  
%     pause(0.01)
%     clf
% end
% close(h)

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


%% 3.4
%
m=0.5;
x2=linspace(-1,1,50);
y2=linspace(-1,1,50);

[X2,Y2]=meshgrid(x2,y2);
l=0.5;

sz=size(X2);
Z=zeros(sz(1),sz(2));
check1=0;
check2=0;
check3=0;
w= waitbar(0,'Please wait...');
for k=1:sz(1)
    msg=num2str(k);
    for j=1:sz(2)
        msg1=num2str(j);
        msg1=strcat(msg1,' out of 20');
        str=strcat(msg,' out of 20 / ');
        str=strcat(str,msg1);
        waitbar(k/sz(1),w,str);
        if sqrt(X2(k,j)^2 + Y2(k,j)^2) > 2*l    %Needs to be on the circunference
            Z(k,j)=NaN;
        else
            %implement theoretical function and get position y1
            aux0=X2(k,j)^2 + Y2(k,j)^2; 
            aux1=Y2(k,j)^2/X2(k,j)^2; 
            aux2= Y2(k,j)/(X2(k,j)^2); 
            y_pos0= roots([(1+aux1) (-(aux0*aux2)) (((aux0^2)/(4*X2(k,j)^2))-l^2)]); 
            y_pos1=max(y_pos0);
            %get x and angles 1 and 2
            x_pos1= (aux0-2*y_pos1*Y2(k,j))/(2*X2(k,j));
            
            theta1i=get_angle1(x_pos1,y_pos1);
            theta2i=get_angle2(x_pos1,y_pos1,X2(k,j),Y2(k,j));
            
%             theta1i= asin(x_pos1/l)
%             theta2i= asin((X2(k,j)-x_pos1)/l)
            %get pi conditions from theoretical formulas
            p1i=(1/6)*m*l^2*(8*0 + 3*(-pi/6)*cos(theta1i-theta2i));
            p2i=(1/6)*m*l^2*(2*(-pi/6));
            
            %we dont need a small fixed step for this problem
            set_simulation_parameters(1,0.5,9.8,theta1i,theta2i,p1i,p2i,'250','0.1',file,'0','0','0','0');  %dont care about position int this function for this case
            mod=sim(file,'SimulationMode','Normal');
            theta1=mod.get('theta1');
            theta2=mod.get('theta2');
            clk=mod.get('clock');
            sze=size(theta1);
            %Simulation time! See if they loop else NaN
            
            ok=0;
            first=0;
            for var=1: sze(1)
                if ((theta1(var)>theta1i+2*pi) || (theta1(var)<theta1i-2*pi) || (theta2(var)>theta2i+2*pi)|| (theta2(var)<theta2i-2*pi))
                    if first==0
                        Z(k,j)=clk(var);
                        Z(k,j)
                        first=1;
                    end
%                     stop=1;
                    %check the timings requested to plot (plot only once for each)
                    if clk(var) > 0 && clk(var) < 30 && check1 == 0
                        figure
                        plot(clk(1:350),theta1(1:350),clk(1:350),theta2(1:350),'linewidth',2);
                        hold on 
                        if (theta1(var)>theta1i+2*pi) || (theta1(var)<theta1i-2*pi)
                            line1=theta1i+2*pi;
                            line2=theta1i-2*pi;
                        else
                            line1=theta2i+2*pi;
                            line2=theta2i-2*pi;
                        end
                        plot(clk(1:350),ones([350,1]) * line1,'r','linewidth',2);
                        plot(clk(1:350),ones([350,1]) * line2,'r','linewidth',2);
                        xlabel('Time (s)');
                        ylabel('(\theta_1, theta_2) [rad]','Interpreter','tex');
                        title(['Loopng in the interval [0, 30](s) for \theta_1(0)=' num2str(theta1i) 'and \theta_2(0)=' num2str(theta2i)],'Interpreter','tex');
                        %legend('N1 given by file','N1 by approximation');
                        grid on
                        grid minor
                        hold off
                        check1=1;
                        break
                    end
                    if clk(var) > 30 && clk(var) < 100 && check2 == 0 && ok==0  %check interval if there is any loop there
                        for t=var:sze(1)
                            theta_g=theta1(var);
                            theta_g2=theta2(var);
                            if (theta1(t)>theta_g+2*pi) || (theta1(t)<theta_g-2*pi)
                                line1=theta_g+2*pi;
                                line2=theta_g-2*pi;
                                lp1=ones([1050,1])*line1;
                                lp2=ones([1050,1])*line2;
                                figure
                                plot(clk(250:1050),theta1(250:1050),clk(250:1050),theta2(250:1050),'linewidth',2);
                                hold on 
                                plot(clk(250:1050),lp1(250:1050),'r',clk(250:1050),lp2(250:1050),'r','linewidth',2);
                                xlabel('Time (s)');
                                ylabel('(\theta_1, theta_2) [rad]','Interpreter','tex');
                                title('Loopng in the interval [30, 100] (s)','Interpreter','tex');
                                %legend('N1 given by file','N1 by approximation');
                                grid on
                                grid minor
                                hold off
                                check2=1;
                                ok=1;
                                break
                            end
                            if (theta2(t)>theta_g2+2*pi) || (theta2(t)<theta_g2-2*pi)
                                line1=theta_g2+2*pi;
                                line2=theta_g2-2*pi;
                                lp1=ones([1050,1])*line1;
                                lp2=ones([1050,1])*line2;
                                figure
                                plot(clk(250:1050),theta1(250:1050),clk(250:1050),theta2(250:1050),'linewidth',2);
                                hold on 
                                plot(clk(250:1050),lp1(250:1050),'r',clk(250:1050),lp2(250:1050),'r','linewidth',2);
                                xlabel('Time (s)');
                                ylabel('(\theta_1, theta_2) [rad]','Interpreter','tex');
                                title('Loopng in the interval [30, 100] (s)','Interpreter','tex');
                                %legend('N1 given by file','N1 by approximation');
                                grid on
                                grid minor
                                hold off
                                check2=1;
                                ok=1;
                                break
                            end
                            ok=1;
                        end 
                    end
                    if clk(var) > 100 && clk(var) < 250 && check3 == 0 && ok==0 %check interval if there is any loop there
                        for t=var:sze(1)
                            theta_g=theta1(var);
                            theta_g2=theta2(var);
                            if (theta1(t)>theta_g+2*pi) || (theta1(t)<theta_g-2*pi)
                                line1=theta_g+2*pi;
                                line2=theta_g-2*pi;
                                lp1=ones([2499,1])*line1;
                                lp2=ones([2499,1])*line2;
                                figure
                                plot(clk(950:2499),theta1(950:2499),clk(950:2499),theta2(950:2499),'linewidth',2);
                                hold on 
                                plot(clk(950:2499),lp1(950:2499),'r',clk(950:2499),lp2(950:2499),'r','linewidth',2);
                                xlabel('Time (s)');
                                ylabel('(\theta_1, theta_2) [rad]','Interpreter','tex');
                                title('Loopng in the interval [100, 250] (s)','Interpreter','tex');
                                %legend('N1 given by file','N1 by approximation');
                                grid on
                                grid minor
                                hold off
                                check3=1;
                                ok=1;
                                break
                            end
                            if (theta2(var)>theta_g2+2*pi) || (theta2(var)<theta_g2-2*pi)
                                line1=theta_g2+2*pi;
                                line2=theta_g2-2*pi;
                                lp1=ones([2499,1])*line1;
                                lp2=ones([2499,1])*line2;
                                figure
                                plot(clk(950:2499),theta1(950:2499),clk(950:2499),theta2(950:2499),'linewidth',2);
                                hold on 
                                plot(clk(950:2499),lp1(950:2499),'r',clk(950:2499),lp2(950:2499),'r','linewidth',2);
                                xlabel('Time (s)');
                                ylabel('(\theta_1, theta_2) [rad]','Interpreter','tex');
                                title('Loopng in the interval [100, 250] (s)','Interpreter','tex');
                                %legend('N1 given by file','N1 by approximation');
                                grid on
                                grid minor
                                hold off
                                check3=1;
                                ok=1;
                                break
                            end
                            ok=1;
                        end
                    end
                else            %if there is no loop 
                    %if there is no loop in the entire sim. =NaN
                    if first==0
                        Z(k,j)=NaN;
                    end
                end
            end
        end
    end       
end
close(w)
figure
pcolor(X2,Y2,Z)
colorbar;


end

function set_simulation_parameters(m,l,g,th1_init,th2_init,p1_init,p2_init,StopTime,Step,file,x1_init,y1_init,x2_init,y2_init)
    set_param(file,'StopTime',StopTime);
    set_param(file,'FixedStep',Step);
    
    th1_init=num2str(th1_init);
    th2_init=num2str(th2_init);
    p1_init=num2str(p1_init);
    p2_init=num2str(p2_init);

    d_theta1=fcn_theta1(m,l);
    d_theta2= fcn_theta2(m,l);
    d_p1=fcn_p1(m,l,g);
    d_p2=fcn_p2(m,l,g);
    str5=fcn_x1(l);
    str6=fc_y1(l);
    str7=fcn_x2(l);
    str8=fcn_y2(l);
    
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

function angle= get_angle1(x1,y1)
    if x1>0 && y1>0
        angle=atan2((y1),(x1)) + pi/2;
    elseif x1>0 && y1<0 
        angle=atan2((y1),(x1)) + pi/2;
    elseif x1<0 && y1>0
        angle=atan2((y1),(x1)) + pi/2;
    else 
        angle=atan2((y1),(x1)) + 5*pi/2;
    end
    
end

function angle = get_angle2(x1,y1,x2,y2)
    if x2>0 && y2>0
        angle=atan2((y2-y1),(x2-x1)) + pi/2;
    elseif x2>0 && y2<0 
        angle=atan2((y2-y1),(x2-x1)) + pi/2;
    elseif x2<0 && y2>0
        angle=atan2((y2-y1),(x2-x1)) + pi/2;
    else 
        angle=atan2((y2-y1),(x2-x1)) + 5*pi/2;
    end
    
end