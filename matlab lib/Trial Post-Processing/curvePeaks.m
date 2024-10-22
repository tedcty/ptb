function [peaks,knees] = curvePeaks(curves)
%curvePeaks: pull peak values for curves
%Takes time curves from specified structure, and finds peak values
% curves: string name of structure

%%      CompiledCurves
%       is indexed as follows:
%
%       CompiledCurves(num).leftAngle.cond{1,cycle}(:,col)
%       where
%       num : indicates which subject (so AB02 is 1, in this case)
%       cond: indicates which condition (so ss is 1, in this case)
%       cycle:indicates which gait cycle(so 01 is 1, in this case)
%       col : indicates which column (1 is time, then the others are
%       forPlotting [likely hip, knee, ankle])
%
%       For average gait, CompiledCurves(num).Average.cond.side.L_Angle(time,col)

% So for a plotting example, this loop will plot all left hip angles as cycles from AB08
% for i = 1:10
% plot(CompiledCurves(1).leftAngle.ss{1,i}(:,2))
% hold on
% end
for i = 1:length(curves) % loop through participants
    for j =1:length(curves(i).ConditionList) % loop through conditions
        for k = 1:length(curves(i).leftAngle.(curves(i).ConditionList{j}))
            for q = 2:(length(curves(i).joints)+1)
                switch curves(i).joints{q-1}
                    case 'hip_flexion'
                        % Peak Hip Flexion
                        peaks(i).leftAngle.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).leftAngle.(curves(i).ConditionList{j}){1,k}(:,q)); % Minimum Value (Max extension)
                        peaks(i).leftMoment.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).leftMoment.(curves(i).ConditionList{j}){1,k}(:,q)); % Minimum Value (Max extension)
                        peaks(i).leftPower.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).leftPower.(curves(i).ConditionList{j}){1,k}(floor(end/2):end,q)); %  Maximum Value from halfway (H3 Peak)
                    case 'knee_angle'
                        % Peak Knee Flexion
                        peaks(i).leftAngle.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).leftAngle.(curves(i).ConditionList{j}){1,k}(:,q)); % Maximum Value (peak flexion)
                        peaks(i).leftMoment.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).leftMoment.(curves(i).ConditionList{j}){1,k}(floor(0.3*end):floor(0.7*end),q)); % Minimum mid-stride value
                        knees(i).leftMoment.(curves(i).ConditionList{j})(k,:) = max(curves(i).leftMoment.(curves(i).ConditionList{j}){1,k}(floor(0.45*end):end,q)); % Max late-stride value
                        peaks(i).leftPower.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).leftPower.(curves(i).ConditionList{j}){1,k}(floor(0.3*end):floor(0.7*end),q)); % Minimum mid-stride value
                        knees(i).leftPower.(curves(i).ConditionList{j})(k,:) = max(curves(i).leftPower.(curves(i).ConditionList{j}){1,k}(floor(0.3*end):floor(0.7*end),q)); % Minimum mid-stride value
                    case 'ankle_angle'
                        % Peak Ankle Flexion
                        peaks(i).leftAngle.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).leftAngle.(curves(i).ConditionList{j}){1,k}(:,q)); % Minimum Value (Max Plantarflexion)
                        peaks(i).leftMoment.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).leftMoment.(curves(i).ConditionList{j}){1,k}(:,q)); % Maximum Value (Max Plantarflexion Moment)
                        peaks(i).leftPower.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).leftPower.(curves(i).ConditionList{j}){1,k}(:,q)); % Maximum Value (Max Plantarflexion Power)
                    otherwise
                        disp('Not a recognised joint!')
                end
            end
        end
        for k = 1:length(curves(i).rightAngle.(curves(i).ConditionList{j}))
            for q = 2:(length(curves(i).joints)+1)
                switch curves(i).joints{q-1}
                    case 'hip_flexion'
                        % Peak Hip Flexion
                        peaks(i).rightAngle.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).rightAngle.(curves(i).ConditionList{j}){1,k}(:,q)); % Minimum Value (Max extension)
                        peaks(i).rightMoment.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).rightMoment.(curves(i).ConditionList{j}){1,k}(:,q)); % Minimum Value (Max extension)
                        peaks(i).rightPower.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).rightPower.(curves(i).ConditionList{j}){1,k}(floor(end/2):end,q)); %  Maximum Value from halfway (H3 Peak)
                    case 'knee_angle'
                        % Peak Knee Flexion
                        peaks(i).rightAngle.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).rightAngle.(curves(i).ConditionList{j}){1,k}(:,q)); % Maximum Value (peak flexion)
                        peaks(i).rightMoment.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).rightMoment.(curves(i).ConditionList{j}){1,k}(floor(0.3*end):floor(0.7*end),q)); % Minimum mid-stride value
                        knees(i).rightMoment.(curves(i).ConditionList{j})(k,:) = max(curves(i).rightMoment.(curves(i).ConditionList{j}){1,k}(floor(0.45*end):end,q)); % Max late-stride value
                        peaks(i).rightPower.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).rightPower.(curves(i).ConditionList{j}){1,k}(floor(0.3*end):floor(0.7*end),q)); % Minimum mid-stride value
                        knees(i).rightPower.(curves(i).ConditionList{j})(k,:) = max(curves(i).rightPower.(curves(i).ConditionList{j}){1,k}(floor(0.3*end):floor(0.7*end),q)); % Minimum mid-stride value
                    case 'ankle_angle'
                        % Peak Ankle Flexion
                        peaks(i).rightAngle.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = min(curves(i).rightAngle.(curves(i).ConditionList{j}){1,k}(:,q)); % Minimum Value (Max Plantarflexion)
                        peaks(i).rightMoment.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).rightMoment.(curves(i).ConditionList{j}){1,k}(:,q)); % Maximum Value (Max Plantarflexion Moment)
                        peaks(i).rightPower.(curves(i).ConditionList{j}).(curves(i).joints{q-1})(k,:) = max(curves(i).rightPower.(curves(i).ConditionList{j}){1,k}(:,q)); % Maximum Value (Max Plantarflexion Power)
                    otherwise
                        disp('Not a recognised joint!')
                end
            end
        end
    end
end
end

