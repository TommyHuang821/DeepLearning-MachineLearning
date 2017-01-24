clc;clear;close all
mu=[1 5];
sigma=[2 3; 3 5];
data=mvnrnd(mu,sigma,1000);
figure
scatter(data(:, 1), data(:, 2));
xlabel('x_1');ylabel('x_2');
title('Raw data');
hold off

% PCA
[RotMatrix coe_PC xRot]=pca(data');



figure
hold on
% vector of PCs
plot([0 10*RotMatrix(1,1)], [0 10*RotMatrix(1,2)]);
plot([0 10*RotMatrix(2,1)], [0 10*RotMatrix(2,2)]);
% zeros means
[dim n]=size(data');
avg = mean(data);
x = data' - repmat(avg', 1, n);
scatter(x(1, :), x(2, :));
plot([0 20*RotMatrix(1,1)], [0 20*RotMatrix(1,2)],'LineWidth',2);
plot([0 20*RotMatrix(2,1)], [0 20*RotMatrix(2,2)],'LineWidth',2);
xlabel('x_1');ylabel('x_2');
title('Raw data with zero-mean');
set(gca,'xlim',[-5 5],'ylim',[-5 5])
hold off

figure
scatter(xRot(2, :), xRot(1, :));
xlabel('rot PC_1');ylabel('rot PC_2');
title('PCA (Projected data)');

