function [TOframe] = findToeOffs(GRFZ)
TOframe = [];
window = 20;


% New thresholder; previously worked with zero. if necessary, set
% weightThresh to zero.
sortedGRFZ = sort(GRFZ);
weightThresh = mean(sortedGRFZ(end-100:end))/10; %Get the mean of the 100 highest points as maximal loading, times 0.1 for thresholding.

for i = window:length(GRFZ)-window
    if (mean(GRFZ(i-(window-1):i))>weightThresh) && (mean(GRFZ(i+1:i+window))<weightThresh) && (GRFZ(i)<weightThresh+10) && ((GRFZ(i)>weightThresh-10)||(GRFZ(i+1)<10))
        TOframe = [TOframe i];
    end
end

%get rid of extras
diff = TOframe(2:end) - TOframe(1:end-1);
TOframe(diff<100) = [];

end