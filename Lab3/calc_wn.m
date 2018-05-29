function [wn, dt] = calc_wn(beta,k, l, m, L, M)
    gamma1 = l*m + (L/2)*M;
    gamma2 = (1/3)*M*(L.^2) + m *l.^2;
       
     wn = sqrt((k-gamma1*9.8)/gamma2);
     dt = ((beta)/gamma2)*(1/(2*wn));
end