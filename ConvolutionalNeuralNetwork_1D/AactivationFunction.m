function [af daf]=AactivationFunction(ActF)

if strcmp(ActF,'sigmoid')
    af=@(x) (1./(1+exp(-x))); % sigmoid function
    daf=@(x) (af(x)).*(1.-af(x)); % deviated sigmoid function
elseif strcmp(ActF,'linear')
    af=@(x) (x); % linear
    daf=@(x) (1); % deviated linear
elseif strcmp(ActF,'tanh')
    af=@(x) ((exp(x)-exp(-x))./(exp(x)+exp(-x))); % 
    daf=@(x) (1- af(x).^2); % deviated tanh
elseif strcmp(ActF,'ReLU') % Rectified linear unit 
    af=@(x) (double(x>=0).*x); 
    daf=@(x) double(af(x)>=0); % deviated
end
