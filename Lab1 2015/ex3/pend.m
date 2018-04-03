function 

    
%Equations
%eq1 theta1 der
    m=num2str(m);
    l=num2str(l);
    term1=strcat('(6/',)
    term1_N1=strcat(delt1,'*u(1)');
    alpha1=num2str(alpha1);
    aux=strcat('-',alpha1);
    term2_N1=strcat(aux,'*u(1)*u(2)');
    prey_fcn=strcat(term1_N1,term2_N1);

end