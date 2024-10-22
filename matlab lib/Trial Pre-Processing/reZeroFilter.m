function [filterPlate] = reZeroFilter(originalFys)
%reZeroForceData: Resets all values that were originally zero to zero.

for i = 1:length(originalFys(1,:))
    positiveFy = originalFys(:,i) > 20;
    
    trueInd = strfind(positiveFy',[1 1 1 1 1 1 1 1 1 1]);
    extraInd = trueInd(1:end-1) + 10;
    
    filterPlate(:,i) = zeros(length(originalFys(:,i)),1);
    filterPlate(trueInd,i) = 1;
    filterPlate(extraInd,i) = 1;
end