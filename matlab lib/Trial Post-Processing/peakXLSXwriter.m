
loadingbar = waitbar(0,'Writing to xlsx files');
% [~,~,Dem] = xlsread('ParticipantDemographics.xlsx');
fulllength = length(CompiledCurves)*54; %Make Robust
% ConditionList = {'ss','FBHi','FBLo'};
% forPlotting = {'hip_flexion','knee_angle','ankle_angle'};
for k = 1:length(CompiledCurves)
    participants{k} = CompiledCurves(k).Participant;
    row = find(strcmp(Dem,participants{k}));
    Height = Dem{row,3}./1000;
    Weight = Dem{row,4};
    Leg = Dem{row,6};
    Responder = Dem{row,7};
    
    filename = strcat(CompiledCurves(k).Participant,'PeakValues.xlsx');
    delete(filename)
    clear peakArrayHeaders peakArray peakCell fullCell
    
    peakArrayHeaders(1,1) = {'Cycles'};
    aveind = 1;
    for i = 1:length(forPlotting)
        for j = 1:length(ConditionList)
            column = ((i-1)*12)+((j-1)*6)+2; %NOT FINISHED
            cyclesL = length(peaks(k).leftAngle.(ConditionList{j}).(forPlotting{i}));
            cyclesR = length(peaks(k).rightAngle.(ConditionList{j}).(forPlotting{i}));
            switch Leg
                case 'L'
                    peakArrayHeaders(1,column) = {strcat(ConditionList{j},'_',forPlotting{i},'_','PareticAngle')};
                    peakArray(1:cyclesL,column) = peaks(k).leftAngle.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).leftAngle.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+1) = {strcat(ConditionList{j},'_',forPlotting{i},'_','PareticMoment')};
                    peakArray(1:cyclesL,column+1) = peaks(k).leftMoment.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).leftMoment.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+2) = {strcat(ConditionList{j},'_',forPlotting{i},'_','PareticPower')};
                    peakArray(1:cyclesL,column+2) = peaks(k).leftPower.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).leftPower.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+3) = {strcat(ConditionList{j},'_',forPlotting{i},'_','NonPareticAngle')};
                    peakArray(1:cyclesR,column+3) = peaks(k).rightAngle.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).rightAngle.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+4) = {strcat(ConditionList{j},'_',forPlotting{i},'_','NonPareticMoment')};
                    peakArray(1:cyclesR,column+4) = peaks(k).rightMoment.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).rightMoment.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+5) = {strcat(ConditionList{j},'_',forPlotting{i},'_','NonPareticPower')};
                    peakArray(1:cyclesR,column+5) = peaks(k).rightPower.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).rightPower.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                case 'R'
                    peakArrayHeaders(1,column) = {strcat(ConditionList{j},'_',forPlotting{i},'_','PareticAngle')};
                    peakArray(1:cyclesR,column) = peaks(k).rightAngle.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).rightAngle.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+1) = {strcat(ConditionList{j},'_',forPlotting{i},'_','PareticMoment')};
                    peakArray(1:cyclesR,column+1) = peaks(k).rightMoment.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).rightMoment.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+2) = {strcat(ConditionList{j},'_',forPlotting{i},'_','PareticPower')};
                    peakArray(1:cyclesR,column+2) = peaks(k).rightPower.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).rightPower.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+3) = {strcat(ConditionList{j},'_',forPlotting{i},'_','NonPareticAngle')};
                    peakArray(1:cyclesL,column+3) = peaks(k).leftAngle.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).leftAngle.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+4) = {strcat(ConditionList{j},'_',forPlotting{i},'_','NonPareticMoment')};
                    peakArray(1:cyclesL,column+4) = peaks(k).leftMoment.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).leftMoment.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,column+5) = {strcat(ConditionList{j},'_',forPlotting{i},'_','NonPareticPower')};
                    peakArray(1:cyclesL,column+5) = peaks(k).leftPower.(ConditionList{j}).(forPlotting{i});
                    averagePeaks(k,aveind) = mean(peaks(k).leftPower.(ConditionList{j}).(forPlotting{i})); aveind = aveind + 1;
                    
                otherwise
                    disp('ParticipantDemographics must be correctly populated')
                    break;
            end
            waitbar((column*k)/fulllength,loadingbar);
        end
    end
    kneeColumn = column+6;
    for j = 1:length(ConditionList)
            cyclesL = length(knees(k).leftMoment.(ConditionList{j}));
            cyclesR = length(knees(k).rightMoment.(ConditionList{j}));
            switch Leg
                case 'L'
                    peakArrayHeaders(1,kneeColumn) = {strcat(ConditionList{j},'_knee_angle_','PareticKEM')};
                    peakArray(1:cyclesL,kneeColumn) = knees(k).leftMoment.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).leftMoment.(ConditionList{j})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,kneeColumn+1) = {strcat(ConditionList{j},'_knee_angle_','PareticKPP')};
                    peakArray(1:cyclesL,kneeColumn+1) = knees(k).leftPower.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).leftPower.(ConditionList{j})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,kneeColumn+2) = {strcat(ConditionList{j},'_knee_angle_','NonPareticKEM')};
                    peakArray(1:cyclesR,kneeColumn+2) = knees(k).rightMoment.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).rightMoment.(ConditionList{j})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,kneeColumn+3) = {strcat(ConditionList{j},'_knee_angle_','NonPareticKPP')};
                    peakArray(1:cyclesR,kneeColumn+3) = knees(k).rightPower.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).rightPower.(ConditionList{j})); aveind = aveind + 1;
                    
                case 'R'
                    peakArrayHeaders(1,kneeColumn) = {strcat(ConditionList{j},'_knee_angle_','PareticKEM')};
                    peakArray(1:cyclesR,kneeColumn) = knees(k).rightMoment.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).rightMoment.(ConditionList{j})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,kneeColumn+1) = {strcat(ConditionList{j},'_knee_angle_','PareticKPP')};
                    peakArray(1:cyclesR,kneeColumn+1) = knees(k).rightPower.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).rightPower.(ConditionList{j})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,kneeColumn+2) = {strcat(ConditionList{j},'_knee_angle_','NonPareticKEM')};
                    peakArray(1:cyclesL,kneeColumn+2) = knees(k).leftMoment.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).leftMoment.(ConditionList{j})); aveind = aveind + 1;
                    
                    peakArrayHeaders(1,kneeColumn+3) = {strcat(ConditionList{j},'_knee_angle_','NonPareticKPP')};
                    peakArray(1:cyclesL,kneeColumn+3) = knees(k).leftPower.(ConditionList{j});
                    averagePeaks(k,aveind) = mean(knees(k).leftPower.(ConditionList{j})); aveind = aveind + 1;
                otherwise
                    disp('ParticipantDemographics must be correctly populated')
                    break;
            end
            kneeColumn = kneeColumn+4;
    end
    peakArray(:,1) = 1:length(peakArray(:,1));
    if k == 1
        backupHeaders = peakArrayHeaders;
    end
    peakCell = num2cell(peakArray);
    fullCell = [peakArrayHeaders;peakCell];
    xlswrite(filename,fullCell);
end
filename = 'MeanPeakValues.xlsx';
aveCell = [participants',num2cell(averagePeaks)];
averagepeaksCell = [backupHeaders;aveCell];
xlswrite(filename,averagepeaksCell);
close(loadingbar)