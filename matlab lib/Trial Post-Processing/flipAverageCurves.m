% ConditionList = {'ss','FBHi','FBLo'};
% ConditionList = {'threems','sixms','ninems'};
% forPlotting = {'hip_flexion','knee_angle','ankle_angle'};

for k = 1:length(CompiledCurves)
    if CompiledCurves(k).averageFlipped ~= 1
        for j = 1:length(ConditionList)
            for q = 2:(length(CompiledCurves(k).joints)+1)
                switch CompiledCurves(k).joints{q-1}
                    case 'hip_flexion'
                        CompiledCurves(k).Average.(ConditionList{j}).Left.L_Moment(:,q) = -1.* CompiledCurves(k).Average.(ConditionList{j}).Left.L_Moment(:,q);
                        CompiledCurves(k).Average.(ConditionList{j}).Right.R_Moment(:,q) = -1.* CompiledCurves(k).Average.(ConditionList{j}).Right.R_Moment(:,q);
                    case 'knee_angle'
                        CompiledCurves(k).Average.(ConditionList{j}).Left.L_Angle(:,q) = -1.* CompiledCurves(k).Average.(ConditionList{j}).Left.L_Angle(:,q);
                        CompiledCurves(k).Average.(ConditionList{j}).Right.R_Angle(:,q) = -1.* CompiledCurves(k).Average.(ConditionList{j}).Right.R_Angle(:,q);
                    case 'ankle_angle'
                        CompiledCurves(k).Average.(ConditionList{j}).Left.L_Moment(:,q) = -1.* CompiledCurves(k).Average.(ConditionList{j}).Left.L_Moment(:,q);
                        CompiledCurves(k).Average.(ConditionList{j}).Right.R_Moment(:,q) = -1.* CompiledCurves(k).Average.(ConditionList{j}).Right.R_Moment(:,q);
                end
            end
        end
        CompiledCurves(k).averageFlipped = 1;
    else
        disp('This data has already been flipped!')
    end
end