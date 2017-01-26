clc;clear;close all
label=[];
% genergate Class 1 data
N1=100; % number of class 1 is 100
mu=[1 5];
sigma=[2 1; 1 5];
data1=mvnrnd(mu,sigma,N1);
label=[label;ones(N1,1)];

N2=100; % number of class 2 is 100
mu=[2 0];
sigma=[2 2; 2 5];
data2=mvnrnd(mu,sigma,N2);
label=[label;ones(N2,1)*2];
data=[data1;data2];

figure(1)
for i=1:2
    scatter(data((label==i), 1), data((label==i), 2));
    hold on
    xlabel('x_1');ylabel('x_2');
    title('Raw data');
end
set(gca,'xlim',[-10 10],'ylim',[-10 10])


% DAFE
[RotMatrix, ~, DAFE_dataRot] = dafe(data,label, 2);
[RotMatrix_PC,~,PCA_dataRot] = pca(data');PCA_dataRot=PCA_dataRot';

figure(2)
hold on
for i=1:2
    scatter(data((label==i), 1), data((label==i), 2));
    hold on
end
plot([0 10*RotMatrix(1,1)], [0 10*RotMatrix(1,2)],'k','LineWidth',2);
plot([0 10*RotMatrix(2,1)], [0 10*RotMatrix(2,2)],'k','LineWidth',2);
plot([0 10*RotMatrix_PC(1,1)], [0 10*RotMatrix_PC(1,2)],'r','LineWidth',2);
plot([0 10*RotMatrix_PC(2,1)], [0 10*RotMatrix_PC(2,2)],'r','LineWidth',2);
xlabel('x_1');ylabel('x_2');
title('Raw data with DAFE vector (black) and PCA vector (red) ');
set(gca,'xlim',[-10 10],'ylim',[-10 10])



figure(3)
subplot(2,1,1)
for i=1:2
    scatter(DAFE_dataRot((label==i), 1), DAFE_dataRot((label==i), 2));
    hold on
    xlabel('DAFE vec1');ylabel('DAFE vec2');
    title('DAFE (Projected data)');
end
set(gca,'xlim',[-10 10],'ylim',[-10 10])

subplot(2,1,2)
for i=1:2
    scatter(PCA_dataRot((label==i), 1), PCA_dataRot((label==i), 2));
    hold on
    xlabel('PCA vec1');ylabel('PCA vec2');
    title('PCA (Projected data)');
end
set(gca,'xlim',[-10 10],'ylim',[-10 10])
