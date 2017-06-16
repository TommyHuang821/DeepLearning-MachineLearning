function [af, daf]=AactivationFunction(ActF,option)
% 1. linear
% 2. sigmoid
% 3. tanh
% 4. ReLU: Rectified linear unit 
% 5. LeakyReLU: Leaky rectified linear unit (Leaky ReLU)
% 6. PReLU :  Parameteric rectified linear unit (PReLU)
% 7. ELU : Exponential linear unit (ELU)

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
elseif strcmp(ActF,'LeakyReLU') % Leaky rectified linear unit (Leaky ReLU)
    af=@(x) ((double(x<0)*0.01.*x) + (double(x>=0).*x));
    daf=@(x) (double(af(x)<0)*0.01+double(af(x)>=0)); % deviated
elseif strcmp(ActF,'PReLU') % Parameteric rectified linear unit (PReLU) 
    if nargin<2
        a=0.01;
    else
        a=option;
    end
    af=@(x) ((double(x<0)*a.*x) + (double(x>=0).*x));
    daf=@(x) (double(af(x)<0)*a+double(af(x)>=0)); % deviated
elseif strcmp(ActF,'ELU') % Exponential linear unit (ELU)
    if nargin<2
        a=0.01;
    else
        a=option;
    end
    af=@(x) ((double(x<0)*a.*(exp(x)-1)) + (double(x>=0).*x));
    daf=@(x) (double(af(x)<0).*(a+af(x))+double(af(x)>=0)); % deviated
end
