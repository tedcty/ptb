function [HSframe] = findHeelStrikes(GRFZ)
HSframe = [];
window = 20;

% New thresholder; previously worked with zero. if necessary, set
% weightThresh to zero.
sortedGRFZ = sort(GRFZ);
weightThresh = mean(sortedGRFZ(end-100:end))/10; %Get the mean of the 100 highest points as maximal loading, times 0.1 for thresholding.

for i = (window+1):length(GRFZ)-window
    if (mean(GRFZ(i-window:i-1))<weightThresh) && (mean(GRFZ(i:i+window))>weightThresh) && ((GRFZ(i)<weightThresh+10)||(GRFZ(i-1)<10)) && (GRFZ(i)>weightThresh-10)
        HSframe = [HSframe i];
    end
end

%get rid of extras
diff = HSframe(2:end) - HSframe(1:end-1);
HSframe(diff<100) = [];

end