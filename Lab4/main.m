%% Project by Dinis Rodrigues nº79089 and José Fernandes nº82414
%  For the 1st Laboratory of MSIM     


%% Ex2.a)
% 
function main
close all

%load file
var=load('MarkovChain.mat');
P=var.P;
%get eigenvalues and vectors
[V,D]=eig(P');

for i=1:20
    for j=1:20
        if i==j
           vec=[i abs(1-D(i,j))];  
        end
    end
end

[M,I]=min(vec); %return index of eigenvector corresponding to the eigenvalue=1

%transpose matrix for better understanding
V_t=V(:,I)'; %find vector by D=1
%theoretical equation for normalization
norm=sum(V_t);
v_param=1/norm;
V_norm=V_t*v_param;

%plot figure
figure
bar(V_norm)
grid on;grid minor
xlabel('Probability','Interpreter','Latex');
ylabel('State','Interpreter','Latex');
title('State Probability','Interpreter','Latex');
xlim([0 21])

%prove the sum is 1
figure
bar(sum(V_norm))
xlim([0 2])
xlabel('Graph','Interpreter','Latex');
ylabel('Probability','Interpreter','Latex');
title('Probability Sum','Interpreter','Latex');
[M1,I1]=max(V_norm);
[M2,I2]=min(V_norm);
maximo=num2str(I1);
minimo=num2str(I2);
display(strcat('The more likely state to happen is: ',maximo))
display(strcat('The least likely state to happen is: ',minimo))


%% Ex2.b)
% 

%given values
sigma2=0.01; %deviation
QP=sigma2;
P0=100;         
%from file
anchor_X1=var.nodePos(:,2);
anchor_X2=var.nodePos(:,3);
anchors=[anchor_X1 anchor_X2]';

token_X=var.sourcePos';
X=[token_X zeros(size(token_X)) anchors]';

D=pdist(X,'euclidean');
square=squareform(D); %norm values -> ||x-ai||

distance = square(1,3:end);			% Source-anchor distances -> ||x-ai||^2
anch_norm = square(2,3:end);			% Anchor norms -> ai

n_trial=1000;     %professor asked for 1000 trials
n_anchor=V_norm*n_trial;
n_anchor=round(n_anchor);

num=1;
for i=1:20
   for j=1:n_anchor(1,i)
       di(num)=distance(i);
       a(:,num)=[anchors(1,i), anchors(2,i)];
       an(num)=anch_norm(i);
       num=num+1;
   end
end

Pot = P0./(di.^2);				% Noiseless RSSI
stdev = 1e-1;				% Log-noise standard deviation
%stdev = 0;
Pot = Pot.*exp(stdev*randn(size(Pot)));	% Introduce noise
Pot = QP*round(Pot/QP);			% Quantize power measurements
n = 2;					% Embedding dimension

% Localize source by least-squares
A = [-2*repmat(Pot,[n 1]).*a; -ones(size(Pot)); Pot]';
b = (-Pot.*(an.^2))';

% RLS formulation (one-shot)
RlsPar = struct('lam',1);
[e,w,RlsPar] = qrrls(A,b,RlsPar);

figure
scatter(anchors(1,:), anchors(2,:));
hold on
%plot real position
plot(token_X(1), token_X(2), 'rx','linewidth',2);
%plot measured position
plot(w(1),w(2), 'cd','linewidth',2);
grid on; grid minor;
hold off
%plot description
title('Source and Anchors position','Interpreter','Latex')
xlabel('x1','Interpreter','Latex');
ylabel('x2','Interpreter','Latex');
legend('Anchors Position','Real Source Position','Measured Source Position','Location','northoutside')


%plot a close look
figure
%plot real position
plot(token_X(1), token_X(2), 'rx','linewidth',2);
hold on
%plot measured position
plot(w(1),w(2), 'd','linewidth',2);
grid on; grid minor;
ylim([29 32])
xlim([83 86])
legend('Real Position','Measured Position')
title('Real and Measured Source position ','Interpreter','Latex')
xlabel('x1','Interpreter','Latex');
ylabel('x2','Interpreter','Latex');
hold off

fprintf('Real Source Position -> (%.2f,%.2f) \n',token_X(1), token_X(2));
fprintf('Source Measured Position -> (%.2f,%-2f) \n',w(1),w(2));


%% Ex2.c)
% 

%make minitial conditions transponsed
Pi_init=[randfixedsum(20,1,1,0,1) randfixedsum(20,1,1,0,1) randfixedsum(20,1,1,0,1)]'; %see external functions
Pi_fin=zeros(20,3);

%set time
time=100;
%set variables
X = repmat((1:time)',[1 20]);
Y = repmat(1:20,[time 1]);
P=var.P;
for i=1:3
   Pi=zeros(time,20);
   %set initial conditions
   Pi(1,:)=Pi_init(i,:);
   %theorical equation
   for j=2:time
        Pi(j,:)=Pi(j-1,:)*P;
   end
   %Plot 3d figure
   figure
   plot3(X,Y,Pi);
   grid on; grid minor;
   Pi_fin(:,i)=Pi(100,:);
   
   aux=strcat('Probability Evolution (',num2str(i),')');
   title(aux,'Interpreter','Latex');
   xlabel('Time','Interpreter','Latex');
   ylabel('State','Interpreter','Latex');
   zlabel('Probability','Interpreter','Latex');
   zlim([0 0.2])
end
Sum=[sum(Pi_init(1,:)) sum(Pi_init(2,:)) sum(Pi_init(3,:))];
figure
bar(Sum)
ylim([0 1.2])
title('Probability Sum of the different graphs','Interpreter','Latex');
xlabel('Probability Evolution Graph','Interpreter','Latex');
ylabel('Probability','Interpreter','Latex');

%%
% Given the fact that the initial conditions must be stochastic and valid.
% We made 3 different sets of initial conditions where all of them had a total
% probability of 1. We can see that no matter the initial condition they
% all tend to the equilibrium point.

%% Ex2.d)
% 


end



%% External Functions
% 

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
