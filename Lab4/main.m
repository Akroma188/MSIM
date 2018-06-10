%% Project by Dinis Rodrigues nº79089 and José Fernandes nº82414
%  For the 4th Laboratory of MSIM     


%% Ex2.a)
% 
function main
close all

%load file
var=load('MarkovChain.mat');
P=var.P;
%get eigenvalues and vectors
[V,D]=eig(P');

vec=zeros(1,20);
for i=1:20
   j=i;
   vec(i)=abs(1-D(i,j));      
end

[M,I]=min(vec); %return index of eigenvector corresponding to the eigenvalue closer to 1

%transpose matrix for better understanding
V_t=V(:,I)'; %find vector by D=1
%theoretical equation for normalization
norm=sum(V_t);
v_param=1/norm;
V_norm=V_t*v_param;

%plot figure
figure
bar(V_norm)
grid on;grid minor;
xlabel('State','Interpreter','Latex');
ylabel('Probability','Interpreter','Latex');
title('Equilibrium State Probability','Interpreter','Latex');
xlim([0 21])

%most and less likely to happen
display('The more likely state to happen is: 7 and 19');
display('The least likely state to happen is: 8 and 17');


%% Ex2.b) Markov
% 

%from file
source_X=var.sourcePos';
anchor_X1=var.nodePos(:,2);
anchor_X2=var.nodePos(:,3);
anchors=[anchor_X1 anchor_X2]';

%given values
sigma2=0.01; %deviation
QP=sigma2;
P0=100;   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%from rssiloc script
X=[source_X zeros(size(source_X)) anchors]';

D=pdist(X,'euclidean');
all_norms=squareform(D); %norm values -> ||x-ai||

distance = all_norms(1,3:end);			% Source-anchor distances -> ||x-ai||^2
anch_norm = all_norms(2,3:end);			% Anchor norms -> ||ai||
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%number of times it has been on the anchor
n_anchor=round(V_norm*1000); %professor asked for M=1000 trials and it needs to be an integer

%For Theoretical equation 
di=[];
a=[];
an=[];
%make arrays with M measures
for i=1:20
   di_aux=ones(1,n_anchor(i))*distance(i);
   di=horzcat(di,di_aux);            %for power equation
   
   a_aux=ones(2,n_anchor(i));
   a_aux(1,:)=a_aux(1,:)*anchors(1,i);
   a_aux(2,:)=a_aux(2,:)*anchors(2,i);
   a=horzcat(a,a_aux);              %for A matrix
   
   an_aux=ones(1,n_anchor(i))*anch_norm(i); %for b matrix
   an=horzcat(an,an_aux);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%from rssiloc script
Pot = P0./(di.^2);				% Noiseless RSSI
stdev = 1e-1;				% Log-noise standard deviation

Pot = Pot.*exp(stdev*randn(size(Pot)));	% Introduce noise
Pot = QP*round(Pot/QP);			% Quantize power measurements
n = 2;					% Embedding dimension

% Localize source by least-squares
A = [-2*repmat(Pot,[n 1]).*a; -ones(size(Pot)); Pot]';
b = (-Pot.*(an.^2))';

% RLS formulation (one-shot)
RlsPar = struct('lam',1);
[e,w,RlsPar] = qrrls(A,b,RlsPar);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot measured position
figure
%plot real position
plot(source_X(1), source_X(2), 'x','linewidth',2);
hold on
%plot measured position
plot(w(1),w(2), 'x','linewidth',2);
grid on; grid minor;
ylim([29 32])
xlim([83 86])
legend('Real Position','Measured Position')
title('Real and Measured Source position ','Interpreter','Latex')
xlabel('x1','Interpreter','Latex');
ylabel('x2','Interpreter','Latex');
hold off

err=sqrt((source_X(1)-w(1)).^2 + (source_X(2)-w(2)).^2)

sourceX1pos=num2str(source_X(1),4);
sourceX2pos=num2str(source_X(2),4);
toprint1=strcat('Real Source Position -> (',sourceX1pos,', ',sourceX2pos,')');

MsourceX1pos=num2str(w(1),4);
MsourceX2pos=num2str(w(2),4);
toprint2=strcat('Source Measured Position -> (',MsourceX1pos,', ',MsourceX2pos,')        ',toprint1);
display(toprint2);

%Error calc
display(strcat('Measured root mean square error -> ', num2str(err)));

%% Ex2.c)
% 

%make initial conditions transponsed
%see external functions
Uniform=ones(20,1)*1/20;    %save for future analysis
random1=randfixedsum(20,1,1,0,1);%save for future analysis
random2=randfixedsum(20,1,1,0,1);%save for future analysis

init=[Uniform random1 random2]'; 
%%
% We chose to use as initial conditions a uniform probability distribution
% and two other random distributions to show what happens.

time=linspace(1,100,100);
Pi=zeros(length(time),20);
P=var.P;
for i=1:3
   %set initial condition
   Pi(1,:)=init(i,:);
   %theorical equation (see external functions)
   Pi = MarkovEquation(Pi,P,time);
   
    [V,D]=eig(var.P');
    vec=zeros(1,20);
    for j=1:20
       k=j;
       vec(j)=abs(1-D(j,k));      
    end

    [M,I]=min(vec); %return index of eigenvector corresponding to the eigenvalue=1
    %transpose matrix for better understanding
    V_t=V(:,I)'; %find vector by D=1
    %theoretical equation for normalization
    norm=sum(V_t);
    v_param=1/norm;
    V_norm=V_t*v_param;


    if i==1
      uniform_equilibrium=Pi(end,:); %Save equilibrium for future analysis on 3a)
      Normal_eq=V_norm; %save for 3.a)
    end
   
   %Save sum of elements to prove it's equal to 1
   sum_of_elements(:,i)=sum(Pi,2);
   
   %Plot 3d figure
   [Y,X]=meshgrid(linspace(1,20,20),time);
   figure
   plot3(X,Y,Pi);
   grid on; grid minor;
   compare=strcat('Probability Evolution (',num2str(i),')');
   title(compare,'Interpreter','Latex');
   xlabel('Time','Interpreter','Latex');
   ylabel('State','Interpreter','Latex');
   zlabel('Probability','Interpreter','Latex');
   zlim([0 0.2]);
   
   %Plot equilibrium point
   figure
   bar([V_norm; Pi(end,:)]')
   compare=strcat('Equilibrium (',num2str(i),')');
   title(compare,'Interpreter','Latex');
   legend('Theoretical','Measured');
   xlabel('State','Interpreter','Latex');
   ylabel('Probability','Interpreter','Latex');
   xlim([0 21]);
end

%Plot Sum to prove it's always equal to one
figure
subplot(2,2,[1,2]);
plot(sum_of_elements(:,1));
title('Probability Sum through Time (1)','Interpreter','Latex')
xlabel('Time','Interpreter','Latex');
ylabel('Probability','Interpreter','Latex');
grid on; grid minor;

subplot(2,2,3);
plot(sum_of_elements(:,2));
title('Probability Sum (2)','Interpreter','Latex')
xlabel('Time','Interpreter','Latex');
ylabel('Probability','Interpreter','Latex');
grid on; grid minor;

subplot(2,2,4);
plot(sum_of_elements(:,3))
title('Probability Sum(3)','Interpreter','Latex')
xlabel('Time','Interpreter','Latex');
ylabel('Probability','Interpreter','Latex');
grid on; grid minor;

%%
% Given the fact that the initial conditions must be stochastic and valid.
% We made 3 different sets of initial conditions where all of them had a total
% probability of 1. We can see that no matter the initial condition they
% all tend to the equilibrium point.



%% Ex2.d)
% 

%plot subsets
createfigure(anchor_X1,anchor_X2)
title('Clusters','Interpreter','Latex');
%% 
% There are 4 clusters in this figure, we chose only two where there is a
% greater possibility of the token spending more time on it. We chose the
% top left corner cluster and the bottom righ one.
% Both of them have 80% chances of staying in.
% In this first trial we will try to improve the circulation on these two clusters, we need
% to better distribute the transition probabilities of the clusters so that
% the token spens equally more time in every anchor.
%%%%%%%%%%%%%%%%%% Improved  %%%%%%%%%%%%%%%%%%%%%
P=var.P;
%For anchor 1 (add more transition probability to go the the top left cluster)
P(1,6) = 0.3;
P(1,20) = 0.3;
%For anchor 6 (improve trasition probability to get out of the top left cluster)
P(6,1) = 0.3;
P(6,11) = 0.3;
%For anchor 3 (improve trasition probability to to the bottom right cluster)
P(3,12) = 0.5;
P(3,19) = 0.5;
%For anchor 12 (improve trasition probability to get out of the bottom right cluster)
P(12,3) = 0.3;
P(12,8) = 0.3;
%save improved matrix
P_imp=P;
%%
% we felt no need to change connections between anchors, because if we
% distribute envenly the transition probabilities, it will cover all the
% anchors anyway. Only the time it spends on the anchor matters, not the
% connection between them.
P=var.P;
%%%%%%%%%%%%%%%%%% Worsened case  %%%%%%%%%%%%%%%%%%%%%
%For anchor 1 (add more transition probability to go the the top left cluster)
P(1,6) = 0.8;
P(1,7) = 0.1;
P(1,20) = 0.1;
%For anchor 6 (make it more difficult to go to anchor 1, so it stays in the cluster)
P(6,1) = 0.1;
P(6,15) = 0.4;
P(6,11) = 0.5;
%save worsened matrix
P_w=P;
%%
% We are changing the weigth of the connection so it goes to the
% top cluster and stays there for a much longer period of time.
%Convergence Plot
%set variables
for k=1:2
    if k==1
        P_matrix=P_imp;  %improved matrox
    else
        P_matrix=P_w;    %worsened matrix
    end
    Pi=zeros(100,20);
    %set initial condition
    Pi(1,:)=Uniform;
    %theorical equation
    time=linspace(1,100,100);
    Pi = MarkovEquation(Pi,P_matrix,time);
    %Plot 3d figure
    figure
    [Y,X]=meshgrid(linspace(1,20,20),time);
    plot3(X,Y,Pi);
    grid on; grid minor;
    if k==1
        title('Improved Probability Evolution','Interpreter','Latex');
    else
        title('Worsened Probability Evolution','Interpreter','Latex');
    end
    
    xlabel('Time','Interpreter','Latex');
    ylabel('State','Interpreter','Latex');
    zlabel('Probability','Interpreter','Latex');
    zlim([0 0.2])

    
    [V,D]=eig(P_matrix');
    vec=zeros(1,20);
    for i=1:20
       j=i;
       vec(i)=abs(1-D(i,j));      
    end

    [M,I]=min(vec); %return index of eigenvector corresponding to the eigenvalue=1
    %transpose matrix for better understanding
    V_t=V(:,I)'; %find vector by D=1
    %theoretical equation for normalization
    norm=sum(V_t);
    v_param=1/norm;
    V_norm=V_t*v_param;
    
    %Plot measured equilibrium
    figure
    bar([V_norm; Pi(end,:)]')
    grid on;grid minor
    legend('Equilibrium','Measured')
    xlabel('Probability','Interpreter','Latex');
    ylabel('State','Interpreter','Latex');
    state2='Improved State Probability';
    state3='Worsened State Probability';
    if k==1
        title(state2,'Interpreter','Latex');
        Improved_bar=Pi(end,:);
        Improved_eq=V_norm;
    else
        title(state3,'Interpreter','Latex');
        Worsened_bar=Pi(end,:);
        Worsened_eq=V_norm;
    end
    xlim([0 21])
end

%%
% Comparing this plot with the one of 2.a) we can see that this one is much better
% distributed. If we change the distribution of course the convergence evolution will
% change, and that is what we see in the last figure.
% Now we will try harm the distribuition. We will make it so it stays much
% longer within the clusters discussed above.

%%
% We can clearly see that the token is staying in the top left cluster, as
% expected. We changed the weigth of the connections so it stays in the
% cluster.
%%
% Of course the location precision of the source is affected by the
% fluidity of the token circulation. If, for example in this case (worsened)
% the source is located near the bottom cluster, by this example the token
% stays for long periods of time in the top cluster this will
% imply great error in the location estimation.

%% Ex3.a) MonteCarlo
% 

%set variable
p=Uniform; %(uniform distribution)
n_runs=100;     %number of runs, the bigger the beter but takes more time
points=100;  %time -> points to look at inside each monte run

for n_case=1:3
    Monte.anchor_database=zeros(points,20); %saves how many times the token has been in each anchor in each point of time
    Monte.get_error=zeros(n_runs,points); %saves the error
    %get transition probability
    if n_case ==1
        P=var.P;    %normal transition probability
    elseif n_case==2;
        P=P_imp; %improved transition probability
    else
        P=P_w;  %worsened transition probability
    end
    msg=strcat('Current Trial -> ',num2str(n_case),' of 3:');
    %Preform Monte runs
    for current_run=1:n_runs
        RlsPar = struct('lam',1);
        Monte=MonteRun(var,points,p,Monte.anchor_database,all_norms,current_run,RlsPar,Monte.get_error,QP,stdev,P0,P);     
    end
    
    %Save error
        error_evolution=Monte.get_error;
        if n_case ==1
        	monte_equilibrium_error=sum(error_evolution,1)/n_runs;     %save for 3.b)
        elseif n_case==2;
            monte_improved_equilibrium_error=sum(error_evolution,1)/n_runs;        %save for 3.b)
        else
            monte_worsened_equilibrium_error=sum(error_evolution,1)/n_runs;        %save for 3.b)
        end
        
        %save database
        anchor_database=Monte.anchor_database;
    for m=2:points
        anchor_database(m,:)=anchor_database(m,:)+anchor_database(m-1,:);
        anchor_database(m-1,:)=anchor_database(m-1,:)/((m-1)*n_runs);
    end
    anchor_database(end,:)=anchor_database(end,:)/(points*n_runs);
    
    if n_case ==1
        tittle_ev='Normal Probability Evolution';    
        tittle_eq='Normal Equilibrium Anchor Probability';
        prev_bar=Normal_eq;
        markov_bar=uniform_equilibrium;
    elseif n_case==2;
        tittle_ev='Improved Probability Evolution';        
        tittle_eq='Improved Equilibrium Anchor Probability';
        prev_bar=Improved_eq;
        markov_bar=Improved_bar;
    else
        tittle_ev='Worsened Probability Evolution';        
        tittle_eq='Worsened Equilibrium Anchor Probability';
        prev_bar=Worsened_eq;
        markov_bar=Worsened_bar;
    end
    
    
    [Y,X]=meshgrid(1:20,1:points);
    %Plot Evolution
    figure
    plot3(X,Y,anchor_database);
    grid on; grid minor; zlim([0 0.2])
    title(tittle_ev,'Interpreter','Latex');
    xlabel('Time','Interpreter','Latex');
    ylabel('State','Interpreter','Latex');
    zlabel('Probability','Interpreter','Latex');
    %Plot Equilibrium
    figure
    compare=[prev_bar; markov_bar; anchor_database(end,:)]';
    bar(compare);
    title(tittle_eq,'Interpreter','Latex');
    xlabel('Anchor','Interpreter','Latex');
    ylabel('Probability','Interpreter','Latex');
    legend('T. Equilibrium','Markov','MonteCarlo')
    if n_case==3
        xlim([0 21]); ylim([0 0.25]);
    else
        xlim([0 21]); ylim([0 0.12]);
    end
end


%% 3.b) 
%

figure
hold on
plot(monte_equilibrium_error)
plot(monte_improved_equilibrium_error)
plot(monte_worsened_equilibrium_error,'k')
grid on;grid minor;
legend('Normal','Improved','Worsened')
title('Error Evolution','Interpreter','Latex')
xlabel('Time','Interpreter','Latex')
ylabel('Error','Interpreter','Latex')

%%
% We can clearly see that if we change the transition probability to a worst case scenario, the
% error will be much greater. By improving the circulation through all the
% anchors we can also notice a slightly accurate estimate. In
% the begining we get those spikes because with few estimates of the
% anchors we cant get a good estimate.
end

%% External Functions
%

%% Markov Equation
% 

%%
% This is simply the theoretical equation to give us the final Pi matrix
function Pi = MarkovEquation(Pi,P,time)
    for j=2:length(time)
        Pi(j,:)=Pi(j-1,:)*P;
    end
end

%% Error Calculation
% 

%%
% In here we add each trial to an array in order to preform an incremental
% error calculation like we did in 2.b). We chose to go by the root square
% mean error.
function MonteS=error_calc(point,n_run,RlsPar,all_norms,curr_anchor,real_pos,P0,QP,stdev,Mfile,MonteS)
       distance = all_norms(1,3:end);			% Source-anchor distances -> ||x-ai||^2
       anch_norm = all_norms(2,3:end);      
       
       anchors=[Mfile.nodePos(curr_anchor,2) ;Mfile.nodePos(curr_anchor,3)];
       
       ai=anch_norm(curr_anchor);
       di=distance(curr_anchor);
       %same method as in 2.b)
       %we need to add values to the array in order to preform the error
       %calculation through time (adding each trial).
       if point==1
           MonteS.di=di;
           MonteS.a=anchors;
           MonteS.an=ai;
       else 
           MonteS.di=horzcat(MonteS.di,di);
           MonteS.a=horzcat(MonteS.a,anchors);
           MonteS.an=horzcat(MonteS.an,ai);
       end
       
       Pot=P0./MonteS.di.^2;
       Pot = Pot.*exp(stdev*randn);
       Pot = QP*round(Pot/QP);
                 
       A = [-2*repmat(Pot,[2 1]).*MonteS.a; -ones(size(Pot)); Pot]';
       b = (-Pot.*(MonteS.an.^2))';
       
       [e,w,RlsPar] = qrrls(A,b,RlsPar);
       
       %error is given by -> error = [rea_x1 real_x2]T - [m_x1 m_x2]T then sqrt(error(1)^2 - error(2)^2)
       
       error_aux=[real_pos(1)-w(1);real_pos(2)-w(2)];
       error=sqrt(error_aux(1).^2 + error_aux(2).^2);
       MonteS.get_error(n_run,point)= error;
end
%% Monte Run
% 

%%
% In here we preform a Monte Run, we go through each point of time and
% calculate wich is the next anchor it's going, we save it an then add it
% to the monte error calculation. Anchor database is the array we need to
% then plot the probability evolution. It also give us the error matrix
% that we then need to plot the error evolution.
function MonteS=MonteRun(MChain_struct,time,init_p,anchor_database,all_norms,n_run,RlsPar,get_error,QP,stdev,P0,P)
    real_pos=MChain_struct.sourcePos'; %for error estimation
    MonteS.di=[];
    MonteS.a=[];
    MonteS.an=[];
    MonteS.get_error=get_error;
    for i=1:time
        if i==1
            first_anchor=find(cumsum(init_p)>rand,1,'first'); %define first anchor
            current_anchor=first_anchor;  %current anchor (defined to simplify code reading)
        else
            next_anchor=find(cumsum(P(current_anchor,:)')>rand,1,'first');
            current_anchor=next_anchor;
        end
        %update database
        anchor_database(i,current_anchor)=anchor_database(i,current_anchor)+1;
        %calculate error
        MonteS=error_calc(i,n_run,RlsPar,all_norms,current_anchor,real_pos,P0,QP,stdev,MChain_struct,MonteS);
        
    end
    
    MonteS.anchor_database=anchor_database;
    
end



%code generated using 'Generate Code' option of figure
function createfigure(X1, Y1)
%CREATEFIGURE(X1, Y1)
%  X1:  vector of x data
%  Y1:  vector of y data

%  Auto-generated by MATLAB on 29-May-2018 23:03:43
% Create figure
figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(X1,Y1,'Marker','o','LineStyle','none');
box(axes1,'on');
annotation(figure1,'line',[0.14609375 0.46484375],...
    [0.858977949283352 0.887541345093716]);
annotation(figure1,'line',[0.14453125 0.2046875],...
    [0.857875413450937 0.825799338478501]);
annotation(figure1,'line',[0.2046875 0.28203125],...
    [0.825901874310915 0.780595369349504]);
annotation(figure1,'line',[0.28203125 0.46640625],...
    [0.777390297684675 0.88864388092613]);
annotation(figure1,'line',[0.46640625 0.421875],...
    [0.886541345093716 0.700110253583241]);
annotation(figure1,'line',[0.421875 0.42734375],...
    [0.698007717750827 0.514884233737597]);
annotation(figure1,'line',[0.428125 0.23671875],...
    [0.514986769570011 0.448732083792723]);
annotation(figure1,'line',[0.2375 0.184375],...
    [0.447732083792723 0.240352811466373]);
annotation(figure1,'line',[0.184375 0.3375],...
    [0.237147739801544 0.24696802646086]);
annotation(figure1,'line',[0.4734375 0.3375],...
    [0.335273428886439 0.248070562293275]);
annotation(figure1,'line',[0.428125 0.47265625],...
    [0.514986769570011 0.337375964718853]);
annotation(figure1,'line',[0.421875 0.23671875],...
    [0.696905181918412 0.448732083792723]);
annotation(figure1,'line',[0.42734375 0.54765625],...
    [0.513884233737597 0.577728776185226]);
annotation(figure1,'line',[0.54765625 0.60859375],...
    [0.575626240352812 0.757442116868798]);
annotation(figure1,'line',[0.6078125 0.7421875],...
    [0.755339581036384 0.835722160970232]);
annotation(figure1,'line',[0.7421875 0.72890625],...
    [0.832517089305402 0.650496141124587]);
annotation(figure1,'line',[0.72890625 0.60859375],...
    [0.650598676957001 0.758544652701213]);
annotation(figure1,'line',[0.703125 0.54765625],...
    [0.477500551267916 0.577728776185226]);
annotation(figure1,'line',[0.72890625 0.5484375],...
    [0.650598676957001 0.575523704520397]);
annotation(figure1,'line',[0.703125 0.628125],...
    [0.477500551267916 0.249173098125689]);
annotation(figure1,'line',[0.63046875 0.6015625],...
    [0.249275633958104 0.143329658213892]);
annotation(figure1,'line',[0.62890625 0.69375],...
    [0.247070562293275 0.253583241455347]);
annotation(figure1,'line',[0.69453125 0.6546875],...
    [0.253685777287762 0.154355016538037]);
annotation(figure1,'line',[0.6546875 0.6015625],...
    [0.152252480705623 0.144432194046307]);
annotation(figure1,'line',[0.69375 0.87109375],...
    [0.251480705622933 0.19845644983462]);
annotation(figure1,'line',[0.87109375 0.65390625],...
    [0.195251378169791 0.155457552370452]);
annotation(figure1,'ellipse',...
    [0.12375 0.723140495867769 0.361875 0.264462809917356],...
    'Color',[0.635294139385223 0.0784313753247261 0.184313729405403],...
    'LineWidth',1);
annotation(figure1,'ellipse',...
    [0.5125 0.504132231404959 0.273125 0.419421487603306],...
    'Color',[0.0784313753247261 0.168627455830574 0.549019634723663],...
    'LineWidth',1);
annotation(figure1,'ellipse',...
    [0.5635 0.0743801652892562 0.3315 0.276859504132232],...
    'Color',[0 0.498039215803146 0],...
    'LineWidth',1);
annotation(figure1,'ellipse',...
    [0.161 0.0495867768595041 0.353375000000001 0.708677685950413],...
    'Color',[0.749019622802734 0 0.749019622802734],...
    'LineWidth',1);
end

%This function was found on the internet to give an n by m array in which the sum
%of all the elements is set by the user
function [x,v] = randfixedsum(n,m,s,a,b)

% [x,v] = randfixedsum(n,m,s,a,b)
%
%   This generates an n by m array x, each of whose m columns
% contains n random values lying in the interval [a,b], but
% subject to the condition that their sum be equal to s.  The
% scalar value s must accordingly satisfy n*a <= s <= n*b.  The
% distribution of values is uniform in the sense that it has the
% conditional probability distribution of a uniform distribution
% over the whole n-cube, given that the sum of the x's is s.
%
%   The scalar v, if requested, returns with the total
% n-1 dimensional volume (content) of the subset satisfying
% this condition.  Consequently if v, considered as a function
% of s and divided by sqrt(n), is integrated with respect to s
% from s = a to s = b, the result would necessarily be the
% n-dimensional volume of the whole cube, namely (b-a)^n.
%
%   This algorithm does no "rejecting" on the sets of x's it
% obtains.  It is designed to generate only those that satisfy all
% the above conditions and to do so with a uniform distribution.
% It accomplishes this by decomposing the space of all possible x
% sets (columns) into n-1 dimensional simplexes.  (Line segments,
% triangles, and tetrahedra, are one-, two-, and three-dimensional
% examples of simplexes, respectively.)  It makes use of three
% different sets of 'rand' variables, one to locate values
% uniformly within each type of simplex, another to randomly
% select representatives of each different type of simplex in
% proportion to their volume, and a third to perform random
% permutations to provide an even distribution of simplex choices
% among like types.  For example, with n equal to 3 and s set at,
% say, 40% of the way from a towards b, there will be 2 different
% types of simplex, in this case triangles, each with its own
% area, and 6 different versions of each from permutations, for
% a total of 12 triangles, and these all fit together to form a
% particular planar non-regular hexagon in 3 dimensions, with v
% returned set equal to the hexagon's area.
%
% Roger Stafford - Jan. 19, 2006

% Check the arguments.
if (m~=round(m))|(n~=round(n))|(m<0)|(n<1)
 error('n must be a whole number and m a non-negative integer.')
elseif (s<n*a)|(s>n*b)|(a>=b)
 error('Inequalities n*a <= s <= n*b and a < b must hold.')
end

% Rescale to a unit cube: 0 <= x(i) <= 1
s = (s-n*a)/(b-a);

% Construct the transition probability table, t.
% t(i,j) will be utilized only in the region where j <= i + 1.
k = max(min(floor(s),n-1),0); % Must have 0 <= k <= n-1
s = max(min(s,k+1),k); % Must have k <= s <= k+1
s1 = s - [k:-1:k-n+1]; % s1 & s2 will never be negative
s2 = [k+n:-1:k+1] - s;
w = zeros(n,n+1); w(1,2) = realmax; % Scale for full 'double' range
t = zeros(n-1,n);
tiny = 2^(-1074); % The smallest positive matlab 'double' no.
for i = 2:n
 tmp1 = w(i-1,2:i+1).*s1(1:i)/i;
 tmp2 = w(i-1,1:i).*s2(n-i+1:n)/i;
 w(i,2:i+1) = tmp1 + tmp2;
 tmp3 = w(i,2:i+1) + tiny; % In case tmp1 & tmp2 are both 0,
 tmp4 = (s2(n-i+1:n) > s1(1:i)); % then t is 0 on left & 1 on right
 t(i-1,1:i) = (tmp2./tmp3).*tmp4 + (1-tmp1./tmp3).*(~tmp4);
end

% Derive the polytope volume v from the appropriate
% element in the bottom row of w.
v = n^(3/2)*(w(n,k+2)/realmax)*(b-a)^(n-1);

% Now compute the matrix x.
x = zeros(n,m);
if m == 0, return, end % If m is zero, quit with x = []
rt = rand(n-1,m); % For random selection of simplex type
rs = rand(n-1,m); % For random location within a simplex
s = repmat(s,1,m);
j = repmat(k+1,1,m); % For indexing in the t table
sm = zeros(1,m); pr = ones(1,m); % Start with sum zero & product 1
for i = n-1:-1:1  % Work backwards in the t table
 e = (rt(n-i,:)<=t(i,j)); % Use rt to choose a transition
 sx = rs(n-i,:).^(1/i); % Use rs to compute next simplex coord.
 sm = sm + (1-sx).*pr.*s/(i+1); % Update sum
 pr = sx.*pr; % Update product
 x(n-i,:) = sm + pr.*e; % Calculate x using simplex coords.
 s = s - e; j = j - e; % Transition adjustment
end
x(n,:) = sm + pr.*s; % Compute the last x

% Randomly permute the order in the columns of x and rescale.
rp = rand(n,m); % Use rp to carry out a matrix 'randperm'
[ig,p] = sort(rp); % The values placed in ig are ignored
x = (b-a)*x(p+repmat([0:n:n*(m-1)],n,1))+a; % Permute & rescale x

end
