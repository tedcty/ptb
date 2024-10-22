function [rightsteps, leftsteps, walkingSpeed] = getStepDisplacements(frames, markerNames, markers, datarate, LHStimes, RHStimes, LTOtimes, RTOtimes) %, LCoP, RCoP, GRFTime
rightsteps = [];
leftsteps = [];

% No longer used for step length, but still for walking speed.
%-------------
RToeInd = find(contains(markerNames,'RToe'));
RHeelInd = find(contains(markerNames,'RCal'));

LToeInd = find(contains(markerNames,'LToe'));
LHeelInd = find(contains(markerNames,'LCal'));

if isempty(RToeInd)
    useHeelR = 1;
else
    useHeelR = 0;
end

if isempty(LToeInd)
    useHeelL = 1;
else
    useHeelL = 0;
end
%-------------

if LHStimes(1) < RHStimes(1)
    leftFirst = 1;
    rightFirst = 0;
else
    leftFirst = 0;
    rightFirst = 1;
end

for i = 1:length(RHStimes)-1
    
    if (i~=1) || (rightFirst == 0)
        HSInd = find(frames(:,2)==RHStimes(i));
        CLTOInd = find(frames(:,2)==LTOtimes(find(LTOtimes>RHStimes(i),1)));
        
        
        startInd = find(frames(:,2)==RHStimes(i));
        endInd = find(frames(:,2)==RHStimes(i+1));
        numFrames = endInd - startInd;
        speedInd(1) = startInd + floor(numFrames*0.15);
        speedInd(2) = startInd + floor(numFrames*0.35);
        timeTaken = (speedInd(2) - speedInd(1))/datarate;
        if (~useHeelR) && (~useHeelL)
            rightsteps(i) = markers(RToeInd).x(HSInd)-markers(LToeInd).x(CLTOInd);
            rspeeds(i) = (markers(RToeInd).x(speedInd(2)) - markers(RToeInd).x(speedInd(1)))/timeTaken;
        else
            rightsteps(i) = markers(RHeelInd).x(HSInd)-markers(LHeelInd).x(CLTOInd);
            rspeeds(i) = (markers(RHeelInd).x(speedInd(2)) - markers(RHeelInd).x(speedInd(1)))/timeTaken;
        end
        
    end
end

if rightsteps(1) == 0
    rightsteps = rightsteps(2:end);
end

for i = 1:length(LHStimes)-1
    
    if (i~=1) || (leftFirst == 0)
        HSInd = find(frames(:,2)==LHStimes(i));
        CLTOInd = find(frames(:,2)==RTOtimes(find(RTOtimes>LHStimes(i),1)));
        
        
        startInd = find(frames(:,2)==LHStimes(i));
        endInd = find(frames(:,2)==LHStimes(i+1));
        numFrames = endInd - startInd;
        speedInd(1) = startInd + floor(numFrames*0.15);
        speedInd(2) = startInd + floor(numFrames*0.35);
        timeTaken = (speedInd(2) - speedInd(1))/datarate;
        if (~useHeelR) && (~useHeelL)
            leftsteps(i) = markers(LToeInd).x(HSInd)-markers(RToeInd).x(CLTOInd);
            lspeeds(i) = (markers(LToeInd).x(speedInd(2)) - markers(LToeInd).x(speedInd(1)))/timeTaken;
        else
            leftsteps(i) = markers(LHeelInd).x(HSInd)-markers(RHeelInd).x(CLTOInd);
            lspeeds(i) = (markers(LHeelInd).x(speedInd(2)) - markers(LHeelInd).x(speedInd(1)))/timeTaken;
        end
    end
end

if leftsteps(1) == 0
    leftsteps = leftsteps(2:end);
end

%average the two legs to be sure
mmspeed = (mean(rspeeds)+mean(lspeeds))/2;

%Round to nearest 50 mm and put into m/s
walkingSpeed = (round(mmspeed/50)*50)/1000;

% If negative, make positive
if walkingSpeed < 0
    walkingSpeed = walkingSpeed * -1;
end

%Check because
% disp(walkingSpeed);

end

%% Original Marker-Based Approach
%
% RToeInd = find(contains(markerNames,'RToe'));
% RHeelInd = find(contains(markerNames,'RCal'));
%
% LToeInd = find(contains(markerNames,'LToe'));
% LHeelInd = find(contains(markerNames,'LCal'));
%
% if isempty(RToeInd)
%     useHeelR = 1;
% else
%     useHeelR = 0;
% end
%
% if isempty(LToeInd)
%     useHeelL = 1;
% else
%     useHeelL = 0;
% end

% for i = 1:length(RHStimes)-1
%     startInd = find(frames(:,2)==RHStimes(i));
%     endInd = find(frames(:,2)==RHStimes(i+1));
%     numFrames = endInd - startInd;
%     speedInd(1) = startInd + floor(numFrames*0.15);
%     speedInd(2) = startInd + floor(numFrames*0.35);
%     timeTaken = (speedInd(2) - speedInd(1))/datarate;
%     rspeedinds(i,1:2) = speedInd;
%     rtimeTakens = timeTaken;
%     if ~useHeelR
%         rightsteps(i) = max(markers(RToeInd).x(startInd:endInd))-min(markers(RToeInd).x(startInd:endInd));
%         rspeed = (markers(RToeInd).x(speedInd(2)) - markers(RToeInd).x(speedInd(1)))/timeTaken;
%         rspeeds(i) = rspeed;
%     else
%         rightsteps(i) = max(markers(RHeelInd).x(startInd:endInd))-min(markers(RHeelInd).x(startInd:endInd));
%         rspeed = (markers(RHeelInd).x(speedInd(2)) - markers(RHeelInd).x(speedInd(1)))/timeTaken;
%         rspeeds(i) = rspeed;
%     end
% end
%
% for i = 1:length(LHStimes)-1
%     startInd = find(frames(:,2)==LHStimes(i));
%     endInd = find(frames(:,2)==LHStimes(i+1));
%     numFrames = endInd - startInd;
%     speedInd(1) = startInd + floor(numFrames*0.15);
%     speedInd(2) = startInd + floor(numFrames*0.35);
%     timeTaken = (speedInd(2) - speedInd(1))/datarate;
%     lspeedinds(i,1:2) = speedInd;
%     ltimeTakens = timeTaken;
%     if ~useHeelL
%         leftsteps(i) = max(markers(LToeInd).x(startInd:endInd))-min(markers(LToeInd).x(startInd:endInd));
%         lspeed = (markers(LToeInd).x(speedInd(2)) - markers(LToeInd).x(speedInd(1)))/timeTaken;
%         lspeeds(i) = lspeed;
%     else
%         leftsteps(i) = max(markers(LHeelInd).x(startInd:endInd))-min(markers(LHeelInd).x(startInd:endInd));
%         lspeed = (markers(LHeelInd).x(speedInd(2)) - markers(LHeelInd).x(speedInd(1)))/timeTaken;
%         lspeeds(i) = lspeed;
%     end
% end
%
% Also CoP Approach
%         HSCoP = RCoP(HSInd,:);
%         CLTOCoP = LCoP(CLTOInd,:);
%
%         if HSCoP(1) == 0 || HSCoP(2) == 0
%             inds = find(RCoP(HSInd+1:HSInd+50,1));
%             HSCoP = RCoP(HSInd+min(inds),:);
%         end
%
%         if CLTOCoP(1) == 0 || CLTOCoP(2) == 0
%             inds = find(LCoP(CLTOInd-50:CLTOInd-1,1));
%             newInd = (CLTOInd-1)-(50-max(inds));
%             HSCoP = LCoP(newInd,:);
%         end

%
%         HSCoP = LCoP(HSInd,:);
%         CLTOCoP = RCoP(CLTOInd,:);
%
%         if HSCoP(1) == 0 || HSCoP(2) == 0
%             inds = find(LCoP(HSInd+1:HSInd+50,1));
%             HSCoP = LCoP(HSInd+min(inds),:);
%         end
%
%         if CLTOCoP(1) == 0 || CLTOCoP(2) == 0
%             inds = find(RCoP(CLTOInd-50:CLTOInd-1,1));
%             newInd = (CLTOInd-1)-(50-max(inds));
%             HSCoP = RCoP(newInd,:);
%         end
%
%         leftsteps = [leftsteps HSCoP(1)-CLTOCoP(1)];
%