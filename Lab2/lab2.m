function lab2

t=linspace(-1.2,1.2,1000);

b=0.5;
y=pB(t,b);
t=(t-0.25)/(0.5/(1+0.5));
plot(t,y);
ylim([0 1.5])

grid on
grid minor

function y=pB(t,b)

    L=length(t);
    y=zeros([1 L]);
    for k=1:L
        if (t(k) > b/2+0.5) || (t(k) < (-b/2)-0.5)
            y(k)=0;
        elseif (t(k) <= 0.5+ b/2) && (t(k) >=0.5)
            y(k)=(4*t(k)^2 -(4*b+4)*t(k) + b^2 + 2*b +1)/(2*b^2);
        elseif (t(k) >= -0.5) && (t(k) <= b/2 - 0.5)
            y(k)=0.5-(4*t(k)^2 +(4-4*b)*t(k) - 2*b +1)/(2*b^2);
        elseif (t(k) >= -b/2-0.5) && (t(k) <= -0.5)
            y(k)=(4*t(k)^2 +(4*b+4)*t(k) + b^2 + 2*b +1)/(2*b^2);
        elseif t(k) >=-b/2+0.5
            y(k)=0.5-(4*t(k)^2 +(4*b-4)*t(k) - 2*b +1)/(2*b^2);
        else
            y(k)=1;
        end
    end
end

end

function yT = U(T,a,b,U1,U2,n1,n2)


end