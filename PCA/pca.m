function [RotMatrix coe_PC xRot ProjectedData]=pca(x)
%  Implement PCA to obtain the rotation matrix (RotMatrix), which is the
%  eigenbasis of covariance of input.
% edit by Chih-Sheng (Tommy) Huang.
%
% Input: 
%       x: input data, dim*n
% Output:
%     RotMatrix: PCA transformation matrix
%     coe_PC: variance 
%     xRot: projected data
%     ProjectedData.xPCAwhite :PCA whitening   
%     ProjectedData.xZCAWhite :ZCA whitening


[dim n]=size(x);
avg = mean(x,2);
x = x - repmat(avg, 1, n);

% covariance matrix
sigma = x * x' / n;
% singular value decomposition (SVD)
[RotMatrix, S] = svd(sigma);
coe_PC=diag(S);
xRot = RotMatrix* x;



% PCA-whitening
epsilon=10^(-5);
xPCAwhite = diag(1./sqrt(coe_PC + epsilon)) * RotMatrix' * x;

% ZCA-whitening
xZCAWhite = RotMatrix * diag(1./sqrt(coe_PC + epsilon)) * RotMatrix' * x;

ProjectedData.xPCAwhite=xPCAwhite;
ProjectedData.xZCAWhite=xZCAWhite;