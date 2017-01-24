function [ Performance C ]=VIndex(x,y)
% Validation Index
%input:
%   x : ture label, must be discrete integer number, for example, 1 2 3 4,...
%   y : predict label
%   
% output:
%   Performance: 
%        first  row : perforamcne  label
%        Second row : accuracy
%        Third row  : kappa coefficient
%        First column: overall peroframnce
%        Second to end column:  per-class peroframnce        
%    --------------------------------------------------------------------
%                     ¢x class 1 (C1) ¢x class 2 (C2) ¢x ... ¢x class n (Cn)  
%    --------------------------------------------------------------------
%    overall accuracy ¢x accuracy(C1) ¢x accuracy(C2) ¢x ... ¢x accuracy(Cn)  
%    ---------------------------------------------------------------------
%    overall kappa    ¢x kappa(C1)    ¢x kappa(C2)    ¢x ... ¢x kappa(Cn)   
%    ---------------------------------------------------------------------
%   C: confusion matrix 

if size(x,1)~=size(y,1)
    y=y';
end
%% confusion matrix
truelabel=max(x) ;
minlabel=min(x) ;
table = tabulate(x);
existlabel=table(:,1);

a=0;
for i=1:length(existlabel)
    a=a+1;
    mis{a}=y(find(x==existlabel(i)));
    b=0; 
    for j=1:length(existlabel)
        b=b+1; 
        C(a,b)=length(find(mis{a}==existlabel(j)));
    end
end
%% testing sample accuracy 
acc=sum(diag(C))/length(y);

%% testing sample accuracy per class
a=0;
for i=1:length(existlabel)
    a=a+1;
    numberperclass(a,1)=length(find(x==existlabel(i)));
end
acc_perclass=diag(C)./numberperclass;

%% kappa coefficient of all class

NormalziedC=C./length(y);
p0=sum(diag(NormalziedC));
c1=sum(NormalziedC,1);
r=sum(NormalziedC,2);
pc=sum(c1.*r');
kappa=(p0-pc)/(1-pc);     

%% kappa coefficient of individual classes
for i=1:length(existlabel)
    kappaperclass(1,i)=(NormalziedC(i,i)-r(i)*c1(i))/(r(i)-r(i)*c1(i));
end

Performance(1,:)=[0,existlabel'];
Performance(2,:)=[acc, acc_perclass'];      
Performance(3,:)=[kappa,  kappaperclass];       