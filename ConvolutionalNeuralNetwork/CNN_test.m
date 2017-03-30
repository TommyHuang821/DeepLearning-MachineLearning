function [OutputLabel Pred ] = CNN_test(CNN_net, test_data)
%  feedforward
CNN_net = CNN_feedforward(test_data, CNN_net);
Pred=CNN_net.Output;
[~,OutputLabel] = max(CNN_net.Output);
% [~, a] = max(test_out);
% bad = find(h ~= a);
% 
% er = numel(bad) / size(test_out, 2);

