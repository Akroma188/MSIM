function pop

 set_param('carro/Integrator','InitialCondition',v0str);
 set_param('carro/Integrator1','InitialCondition',y0str);
 set_param('carro/Gain','Gain',GainStr);
 set_param(file,'StopTime',StopTime);

end