function ExpandMatrix = ExpandMatrix(OriginalMatrix, ExpandSizeVector )
 
SA = size(OriginalMatrix);  % Get the size (and number of dimensions) of input.
T = cell(length(SA), 1);
for ii = length(SA) : -1 : 1
    H = zeros(SA(ii) * ExpandSizeVector(ii), 1);   %  One index vector into A for each dim.
    H(1 : ExpandSizeVector(ii) : SA(ii) * ExpandSizeVector(ii)) = 1;   %  Put ones in correct places.
    T{ii} = cumsum(H);   %  Cumsumming creates the correct order.
end

ExpandMatrix = OriginalMatrix(T{:});   %  Feed the indices into A.