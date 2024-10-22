%ConditionList = {'ss','FBHi','FBLo'};
% ConditionList = {'threems','sixms','ninems'};
% forPlotting = {'hip_flexion','knee_angle','ankle_angle'};
if CompiledCurves(1).flipped ~= 1
    for k = 1:length(CompiledCurves)
        for i = 1:length(forPlotting)
            if strcmp(forPlotting{i},'hip_flexion')
                for j = 1:length(ConditionList)
                    for l = 1:length(CompiledCurves(k).leftMoment.(ConditionList{j}))
                        CompiledCurves(k).leftMoment.(ConditionList{j}){1,l}(:,i+1) = -1.*CompiledCurves(k).leftMoment.(ConditionList{j}){1,l}(:,i+1);
                    end
                    for l = 1:length(CompiledCurves(k).rightMoment.(ConditionList{j}))
                        CompiledCurves(k).rightMoment.(ConditionList{j}){1,l}(:,i+1) = -1.*CompiledCurves(k).rightMoment.(ConditionList{j}){1,l}(:,i+1);
                    end
                end
            elseif strcmp(forPlotting{i},'knee_angle')
                for j = 1:length(ConditionList)
                    for l = 1:length(CompiledCurves(k).leftMoment.(ConditionList{j}))
                        CompiledCurves(k).leftAngle.(ConditionList{j}){1,l}(:,i+1) = -1.*CompiledCurves(k).leftAngle.(ConditionList{j}){1,l}(:,i+1);
                    end
                    for l = 1:length(CompiledCurves(k).rightMoment.(ConditionList{j}))
                        CompiledCurves(k).rightAngle.(ConditionList{j}){1,l}(:,i+1) = -1.*CompiledCurves(k).rightAngle.(ConditionList{j}){1,l}(:,i+1);
                    end
                end
            elseif strcmp(forPlotting{i},'ankle_angle')
                for j = 1:length(ConditionList)
                    for l = 1:length(CompiledCurves(k).leftMoment.(ConditionList{j}))
                        CompiledCurves(k).leftMoment.(ConditionList{j}){1,l}(:,i+1) = -1.*CompiledCurves(k).leftMoment.(ConditionList{j}){1,l}(:,i+1);
                    end
                    for l = 1:length(CompiledCurves(k).rightMoment.(ConditionList{j}))
                        CompiledCurves(k).rightMoment.(ConditionList{j}){1,l}(:,i+1) = -1.*CompiledCurves(k).rightMoment.(ConditionList{j}){1,l}(:,i+1);
                    end
                end
            end
        end
        CompiledCurves(k).flipped = 1;
    end
else
    disp('This data has already been flipped!')
end