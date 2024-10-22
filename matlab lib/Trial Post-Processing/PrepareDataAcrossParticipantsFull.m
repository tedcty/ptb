%% PrepareDataAcrossParticipantsFull
% This script is used to call all related functions to post-process OpenSim
% resultant data for a collection of study participants and trials; the
% structure of said groupings is somewhat hardcoded below, but each
% individual step has been commented to assist in use for whatever data you
% may have.

%% Keep original location for use in File I/O
addpath(pwd)
origFolder = pwd;

%% Debugging Settings
testing = 0; %Change here if you want results not/saved!

if testing == 1
    disp('Results will not be saved!');
end

% currentProblems = {'ProblemSubject1','ProblemSubject2'};

% Plot Heel Strikes and Toe-Offs to check its identifying them correctly.
plotHSTO = 0;

%% Trial Info and Customisations

% Populate Trial Info Structure (SPECIFIC TO SS SET)
populateSSTrials;
demoSheet = 'SSParticipantDemographics.xlsx';
disp('Have you updated the Participant Demographics sheet in the Output Folder?')


% Trim to last ten gait cycles or not (useful if some trials longer than
% others)
trimTen = 1;
firstTen = 0;

% Naming Conventions:
prefix = 'SS'; % Subject Prefix
ConditionList = {'ss','fast'}; % Trial Conditions
forPlotting = {'hip_flexion','knee_angle','ankle_angle'}; % Joint DOFs



% Set in-script ParticipantList
if exist('nameList')
    ParticipantList = nameList;
else
    error('NameList must be provided.')
    % Otherwise
    % nameList = {'Subject1','Subject2'}; etc.
end

% If debugging specific subjects
if exist('currentProblems')
    ParticipantList = currentProblems;
end

% Clear figures
close all; fclose('all');

% Location and Structure
outputFolder = 'C:\Users\dbak576\Desktop\Duncan Bakke Modelling Files\Output';
cd(outputFolder) %navigate to output folder



% OTHER USEFUL TIDBITS
% addpath('C:\Users\dbak576\Desktop\Duncan Bakke Modelling Files');
% ConditionList = {'03ms','06ms','09ms'};
% ConditionNameList = {'threems','sixms','ninems'};
% ConditionList = {'ss','FBHi','FBLo'};
% ParticipantList = {'SS09','SS10','SS21','SS39','SS42','SS48','SS50','SS52','SS58','SS68','SS77','SS78','SS90'};
% OPTIONAL Dynamic Participant List Setting
% OutputList = ls; OutputList(1:2,:) = []; OutputList = cellstr(OutputList); %identify Participants
% if strfind(ConditionList{1},'ms') > 0
%     ParticipantList = {'AB09','AB05','AB02','AB04','AB03','AB06','AB14','AB11','AB08','AB17','AB13','AB15','AB16'};
% else
%     for i=1:length(OutputList)
%         if ~strncmp(OutputList(i),prefix,2)
%             break
%         end
%         if ~strcmp(OutputList{i}(end-3:end),'osim')
%             ParticipantList(j) = OutputList(i);
%             j = j + 1;
%         end
%     end
% end


%% READING IN
[~,~,Dem] = xlsread(demoSheet);
loadingbar = waitbar(0,'Data Preparation');

for q = 1:length(ParticipantList) % Loop through participants
    
    ParticipantName = ParticipantList{q};   % Participant Name (and folder)
    cd(ParticipantName);                    % Go to Participant Directory
    TrialList = ls; TrialList(1:2,:) = [];
    TrialList = cellstr(TrialList);         % List Trials
    
    row = find(strcmp(Dem,ParticipantName));% Go to particpant's row in xlsx
    Height = Dem{row,3}./1000;              % Participant Height and
    Weight = Dem{row,4};                    % weight for normalisation.
    Leg = Dem{row,6};                       % Leg identifier (for unilateral effects)
    normalisingFactor = 1/(Height*Weight);  % Create normalisation factor
    
    for w = 1:length(ConditionList) % Loop through conditions (e.g. ss, fast)
        % Note that this bit is a little ugly, but it works to keep whats
        % necessary and reset others.
        clearvars -except plotHSTO nameList firstTen trimTen currentProblems testing origFolder boxmessage Dem normalisingFactor loadingbar CompiledCurves CompiledData ParticipantList ConditionList ConditionNameList forPlotting TrialList q w ParticipantName ParticipantStruct Leg
        
        rightVertForceID = 'R_ground_force_vy';
        leftVertForceID = 'L_ground_force_vy';
        
        rightCoPxID = 'R_ground_force_px';
        rightCoPzID = 'R_ground_force_pz';
        leftCoPxID = 'L_ground_force_px';
        leftCoPzID = 'L_ground_force_pz';
        
        rightAPForceID = 'R_ground_force_vx';
        leftAPForceID = 'L_ground_force_vx';
        
        %% Data Import
        Condition = ConditionList{w};                               % Set Condition name
        TrialLog = strfind(lower(TrialList),lower([Condition '_']));% Find the corresponding trial in the folder.
        TrialIndex = find(~cellfun(@isempty,TrialLog));
        TrialName = TrialList{TrialIndex};
        cd(TrialName)                                               % Enter Trial Directory (e.g. 27Feb_FBHi_12ms_01)
        
        trcFilename = char([TrialName '.trc']);
        if exist(trcFilename,'file')
            [~, ~, ~, datarate] = readTRC(trcFilename);
        else
            datarate = 200; % You ought to always have a TRC, so this generally doesn't get called. Just for compatibility.
        end
        
        GRFFilename = char([TrialName '.mot']);
        [GRFData{1}, GRFheaderLine(1,:)] = RetrieveMOTData(GRFFilename); %Pull GRF Data from "Trial.mot"
        
        %% Data Framing (Heel Strikes and Toe Offs)
        % This section gets time stamps for gait events during the trial,
        % then trims to area of interest. Note that it rounds these
        % time stamps to the nearest TRC frame.
        
        % Step Time Threshold (steps shorter than this get eliminated)
        threshold = 0.8;
        
        % Special Cases
        % NOTE: These are specific to the SS dataset, but serve as a good
        % example for what you may have to do for your data. Use the
        % plotHSTO section below to check your HS and TO points are being
        % identified properly.
        
        if strcmp(ParticipantName,'SS15')
            threshold = 0.9;
        elseif strcmp(ParticipantName,'SS46')
            threshold = 0.885;
        elseif strcmp(ParticipantName,'SS09')
            threshold = 0.5;
        elseif strcmp(ParticipantName,'SS78')
            firstTen = 1; % An example of a trial that has malformed final steps, so I use early ones instead.
        end

        RZInd = find(strcmp(GRFheaderLine(1,:),rightVertForceID));      % Index in GRF struct corresponding to vertical forces on the right side
        LZInd = find(strcmp(GRFheaderLine(1,:),leftVertForceID));       % Index in GRF struct corresponding to vertical forces on the left side
        GRFTimeInd = find(strcmp(GRFheaderLine(1,:),'time'));           % Index in GRF struct corresponding to time stamps
        GRFTime = GRFData{1}(:,GRFTimeInd);                           	% Time Vector for each data point
        
        R_Z = GRFData{1}(:,RZInd);                                      % Vector of vertical forces on the right side
        RHS_frame = findHeelStrikes(R_Z);                               % HT frames
        RHS_time = round(datarate*GRFData{1}(RHS_frame,1))./datarate;   % HT timestamps
        RHS_TO = findToeOffs(R_Z);                                      % TO frames
        RHS_TO_time = round(datarate*GRFData{1}(RHS_TO,1))./datarate;   % TO timestamps
        
        L_Z = GRFData{1}(:,LZInd);%Vertical Reaction forces on left plate
        LHS_frame = findHeelStrikes(L_Z); % HT frames
        LHS_time = round(datarate*GRFData{1}(LHS_frame,1))./datarate; % HT timestamps
        LHS_TO = findToeOffs(L_Z);% TO frames
        LHS_TO_time = round(datarate*GRFData{1}(LHS_TO,1))./datarate; % TO timestamps

        % Plot without trimming (if needed to debug)
        
        if plotHSTO == 1
            subplot(2,1,1);
            plot(GRFTime,R_Z)
            hold on
            plot(RHS_time,50*ones(length(RHS_time),1),'*b')
            plot(RHS_TO_time,50*ones(length(RHS_TO_time),1),'*r')
            title('Identifed Gait Events (Right)');
            xlabel('Time (s), Heel Strikes (blue), Toe Offs (red)');
            ylabel('Vertical Forces (N)')
            
            subplot(2,1,2);
            plot(GRFTime,L_Z)
            hold on
            plot(LHS_time,50*ones(length(LHS_time),1),'*b')
            plot(LHS_TO_time,50*ones(length(LHS_TO_time),1),'*r')
            title('Identifed Gait Events (left)');
            xlabel('Time (s), Heel Strikes (blue), Toe Offs (red)');
            ylabel('Vertical Forces (N)')
            pause
            close all
        end
        
        % This Participant is a problem case. The 'else' instructions work
        % for every other participant. I've left my manual solution here
        % in case anyone needs it or anything like it.
        if strcmp(ParticipantName,'SS46') && strcmp(Condition,'fast')
            RHS_time_diff = RHS_time(2:end) - RHS_time(1:end-1);
            eliminds = find(RHS_time_diff<threshold)+1;
            RHS_time(eliminds) = []; % Eliminate short frames
            RHS_TO_time_diff = RHS_TO_time(2:end) - RHS_TO_time(1:end-1);
            eliminds = find(RHS_TO_time_diff<threshold)+1;
            RHS_TO_time(eliminds) = []; % Eliminate short frames
            LHS_time_diff = LHS_time(2:end) - LHS_time(1:end-1);
            eliminds = find(LHS_time_diff<threshold)+1;
            LHS_time(eliminds) = []; % Eliminate short frames
            LHS_TO_time_diff = LHS_TO_time(2:end) - LHS_TO_time(1:end-1);
            eliminds = find(LHS_TO_time_diff<threshold)+1;
            LHS_TO_time(eliminds) = []; % Eliminate short frames
        else
            RHS_time_diff = RHS_time(2:end) - RHS_time(1:end-1);    %Find time difference from one event to the next.
            RHS_time(RHS_time_diff<threshold) = [];                 % Eliminate short frames.
            RHS_TO_time_diff = RHS_TO_time(2:end) - RHS_TO_time(1:end-1);
            RHS_TO_time(RHS_TO_time_diff<threshold) = []; % Eliminate short frames
            LHS_time_diff = LHS_time(2:end) - LHS_time(1:end-1);
            LHS_time(LHS_time_diff<threshold) = []; % Eliminate short frames
            LHS_TO_time_diff = LHS_TO_time(2:end) - LHS_TO_time(1:end-1);
            LHS_TO_time(LHS_TO_time_diff<threshold) = []; % Eliminate short frames
        end
        
        % Ensure Toe-Offs are only within existing gait cycles bounded by heel strikes.
        if RHS_TO_time(1)<RHS_time(1)
            RHS_TO_time = RHS_TO_time(2:end);
        end
        
        if LHS_TO_time(1)<LHS_time(1)
            LHS_TO_time = LHS_TO_time(2:end);
        end
        
        if RHS_TO_time(end)>RHS_time(end)
            RHS_TO_time = RHS_TO_time(1:end-1);
        end
        
        if LHS_TO_time(end)>LHS_time(end)
            LHS_TO_time = LHS_TO_time(1:end-1);
        end
        
        %% Trimming Down
        
        if trimTen == 1 % Remove all but 10 gait cycles.
            if length(RHS_time)>10
                if ~firstTen
                    RHS_time = RHS_time(end-10:end);
                    RHS_TO_time = RHS_TO_time(end-9:end);
                else
                    RHS_time = RHS_time(1:11);
                    RHS_TO_time = RHS_TO_time(1:10);
                end
            end
            if length(LHS_time)>10
                if ~firstTen
                    LHS_time = LHS_time(end-10:end);
                    LHS_TO_time = LHS_TO_time(end-9:end);
                else
                    LHS_time = LHS_time(1:11);
                    LHS_TO_time = LHS_TO_time(1:10);
                end
            end
            firstTen = 0;
        end
        
        %% Unused HS/TO Strategies
        
        % Original approach to gait event detection (deprecated by
        % findHeelStrikes and findToeOffs functions), may be useful to
        % someone.
        
        % SS25 and SS78 just have one genuine shuffle each. Eliminate.
        % Original version of exceptions. New TO and HS functions
        % essentially eliminate this.
        %         if strcmp(ParticipantName,'SS15')
        %             threshold = 0.9;
        %             vertThresh = 10;
        %         elseif strcmp(ParticipantName,'SS46')
        %             threshold = 0.88;
        %         elseif strcmp(ParticipantName,'SS09')
        %             threshold = 0.5; %Works for SS09!!!
        %         elseif strcmp(ParticipantName,'SS78')
        %             firstTen = 1;
        %         elseif strcmp(ParticipantName,'SS50')
        %             baddiffthresh = 0.8; %High Variability Steps
        %         end
        
        %         RZInd = find(strcmp(GRFheaderLine(1,:),'R_ground_force_vy'));
        %         LZInd = find(strcmp(GRFheaderLine(1,:),'L_ground_force_vy'));
        %         GRFTimeInd = find(strcmp(GRFheaderLine(1,:),'time'));
        %         GRFTime = GRFData{1}(:,GRFTimeInd);
        
        %         R_Z = GRFData{1}(:,RZInd); %Vertical Reaction forces on right plate
        %         RHS_frame = intersect(find(R_Z>=vertThresh), (1+find(R_Z<vertThresh))); % HT frames
        %         RHS_time = round(datarate*GRFData{1}(RHS_frame,1))./datarate; % HT timestamps
        %         RHS_TO = intersect(find(R_Z<vertThresh), (1+find(R_Z>=vertThresh))); % TO frames
        %         RHS_TO_time = round(datarate*GRFData{1}(RHS_TO,1))./datarate; % TO timestamps

        %         lateDiff = [RHS_TO_time_diff(1); RHS_TO_time_diff];
        %         diffdiffs = lateDiff-[RHS_TO_time_diff; RHS_TO_time_diff(end)];
        %         badInds = find(diffdiffs>baddiffthresh)+1;
        %         RHS_TO_time(badInds) = []; % Eliminate short frames 
        %         RHS_TO_time(RHS_TO_time_diff<threshold) = [];
        
        %         L_Z = GRFData{1}(:,LZInd);%Vertical Reaction forces on left plate
        %         LHS_frame = intersect(find(L_Z>=vertThresh), (1+find(L_Z<vertThresh))); % HT frames
        %         LHS_time = round(datarate*GRFData{1}(LHS_frame,1))./datarate; % HT timestamps
        %         LHS_TO = intersect(find(L_Z<vertThresh), (1+find(L_Z>=vertThresh))); % TO frames
        %         LHS_TO_time = round(datarate*GRFData{1}(LHS_TO,1))./datarate; % TO timestamps
        
        %         lateDiff = [LHS_TO_time_diff(1); LHS_TO_time_diff];
        %         diffdiffs = lateDiff-[LHS_TO_time_diff; LHS_TO_time_diff(end)];
        %         badInds = find(diffdiffs>baddiffthresh)+1;
        %
        %         LHS_TO_time(badInds) = []; % Eliminate short frames
        %         LHS_TO_time(LHS_TO_time_diff<threshold) = [];]
        
        %% Step Lengths
        % Use HS and TO time stamps to find the physical length of each
        % step.
        
        [trcframes, trcmarkerNames, trcmarkers, trcdatarate] = readTRC([TrialName '.trc']); % Read in all TRC data from the trial TRC file
        
        % Get CoP location data from the x and z coordinates
        RCoP(:,1) = GRFData{1}(:,find(strcmp(GRFheaderLine(1,:),rightCoPxID)));
        RCoP(:,2) = GRFData{1}(:,find(strcmp(GRFheaderLine(1,:),rightCoPzID)));
        
        LCoP(:,1) = GRFData{1}(:,find(strcmp(GRFheaderLine(1,:),leftCoPxID)));
        LCoP(:,2) = GRFData{1}(:,find(strcmp(GRFheaderLine(1,:),leftCoPzID)));
        
        % Get step lengths and walking speed.
        [rightsteps, leftsteps, walkingSpeed] = getStepDisplacements(trcframes, trcmarkerNames, trcmarkers, trcdatarate, LHS_time, RHS_time, LHS_TO_time, RHS_TO_time);
        
        
        %% Anterior/Posterior Force Analysis
        
        % Pull the A/P curves
        RXInd = find(strcmp(GRFheaderLine(1,:),rightAPForceID));
        LXInd = find(strcmp(GRFheaderLine(1,:),leftAPForceID));
        R_X = GRFData{1}(:,RXInd); % A/P Reaction forces on right plate
        L_X = GRFData{1}(:,LXInd); % A/P Reaction forces on left plate
        GRFTime = GRFData{1}(:,1);
        
        % Trim to Steps of Interest
        R_X = R_X(find(GRFTime == RHS_time(1)):find(GRFTime == RHS_time(end)));
        L_X = L_X(find(GRFTime == LHS_time(1)):find(GRFTime == LHS_time(end)));
        
        R_X_Positive = R_X;
        L_X_Positive = L_X;
        
        % Get only positive values
        R_X_Positive(R_X_Positive<0) = 0; 
        L_X_Positive(L_X_Positive<0) = 0;
        
        %Get area under Anterior Force curves (positive forward impulse)
        R_Impulse = trapz(R_X_Positive);
        L_Impulse = trapz(L_X_Positive);
        
        % If interested, this allows calculation of "Paretic Propulsion",
        % one of those 'unilateral gait' values I mentioned early.
        if strcmp(Leg,'R')
            pareticPropulsion = R_Impulse / (R_Impulse + L_Impulse);
        else
            pareticPropulsion = L_Impulse / (R_Impulse + L_Impulse);
        end
        
        
        %% Kinematics (Joint Angles)
        
        dataFilename = char([TrialName 'IKResults.mot']);
        [resultantData{1}, headerLine(1,:)] = RetrieveMOTData(dataFilename); % Pull Kinematic Results from "TrialIKResults.mot"
        
        leftSuffix = '_l';
        rightSuffix = '_r';
        
        RCol = 1; % Include Time Column
        LCol = 1; % Include Time Column
        
        for i = 1:length(forPlotting)
            RCol = [RCol find(strcmp(headerLine,[forPlotting{i} rightSuffix]))];    % Identify Right Side Columns of interest
            LCol = [LCol find(strcmp(headerLine,[forPlotting{i} leftSuffix]))];     % Identify Left Side Columns of interest
        end
        
        % Design filter for kinematic data (for more detail on how this
        % works, see MATLAB's "butter" Butterworth filter function, and
        % Winter's "Biomechanics"
        
        Wn=10/(datarate/2);
        [B,A]=butter(2,Wn);
        
        TimeStep = 1/datarate; % Camera rate for velocity
        
        % NOTE: For this section, j corresponds to joint index
                
        % Filter Joint Angle Curves and Populate Data Cells for Analysis
        for i = 1:length(RHS_time)-1                                % Loop through frame time stamps
            startFrame = find(resultantData{1}(:,1)==RHS_time(i));  % Find Start Frame
            endFrame = find(resultantData{1}(:,1)==RHS_time(i+1));	% Find End Frame
            rightKinematicData{i} = filtfilt(double(B),double(A),resultantData{1}(startFrame:endFrame,RCol)); %Filter
            rightKinematicData{i}(:,1) = resultantData{1}(startFrame:endFrame,1); % Time
            for j = 2:length(forPlotting)+1
                rightStanceVel{i}(:,j) = gradient(rightKinematicData{i}(:,j),TimeStep); % Populate Gradients Array (for Power Calculation)
            end
        end
        for i = 1:length(LHS_time)-1 % Same for Left
            startFrame = find(resultantData{1}(:,1)==LHS_time(i));
            endFrame = find(resultantData{1}(:,1)==LHS_time(i+1));
            leftKinematicData{i} = filtfilt(double(B),double(A),resultantData{1}(startFrame:endFrame,LCol));
            leftKinematicData{i}(:,1) = resultantData{1}(startFrame:endFrame,1);
            for j = 2:length(forPlotting)+1
                leftStanceVel{i}(:,j) = gradient(leftKinematicData{i}(:,j),TimeStep);
            end
        end
        
        % Interpolate
        
        rightTOInt = [];
        leftTOInt = [];
        
        % Choose how many points you want between strikes (101 for
        % percentages with zero at initial HS and 100 at end HS)
        interpolationPointNumber = 101;
        
        for i = 1:length(RHS_time)-1
            rightInterpolated{i}(:,1) = linspace(rightKinematicData{i}(1,1),rightKinematicData{i}(end,1),interpolationPointNumber); % Interpolation time stamps
            if i <= length(RHS_TO_time)
                rightTOInt = [rightTOInt ((RHS_TO_time(i)-RHS_time(i))/(RHS_time(i+1)-RHS_time(i))*interpolationPointNumber)]; %If you want Toe Offs
            end
            for j = 1:length(forPlotting)
                rightInterpolated{i}(:,j+1) = interp1(rightKinematicData{i}(:,1),rightKinematicData{i}(:,(1+j)),rightInterpolated{i}(:,1), 'spline'); % Interpolate each joitn curve
            end
        end
        
        for i = 1:length(LHS_time)-1 % Same for Left
            leftInterpolated{i}(:,1) = linspace(leftKinematicData{i}(1,1),leftKinematicData{i}(end,1),interpolationPointNumber);
            if i <= length(LHS_TO_time)
                leftTOInt = [leftTOInt ((LHS_TO_time(i)-LHS_time(i))/(LHS_time(i+1)-LHS_time(i))*interpolationPointNumber)];
            end
            for j = 1:length(forPlotting)
                leftInterpolated{i}(:,j+1) = interp1(leftKinematicData{i}(:,1),leftKinematicData{i}(:,(1+j)),leftInterpolated{i}(:,1), 'spline');
            end
        end
        
        % Package Mean Curves for Output
        percentageValues = 0:1:100; % This MUST be have length of your chosen interpolationPointNumber.
        
        % Percentage Values in first column, preallocate length
        leftTrajectory(:,1) = percentageValues;
        rightTrajectory(:,1) = percentageValues;
        leftStDev(:,1) = percentageValues;
        rightStDev(:,1) = percentageValues;
        rightTrajVector = zeros(length(RHS_time)-1,1);
        leftTrajVector = zeros(length(LHS_time)-1,1);
        
        % Get mean values and standard deviations
        for i = 1:length(percentageValues)
            for j = 1:length(forPlotting)
                for k = 1:(length(RHS_time)-1)
                    rightTrajVector(k) = rightInterpolated{k}(i,j+1);   % Vector of all gait cycle values for joint j at percentage point i
                end
                for k = 1:(length(LHS_time)-1)
                    leftTrajVector(k) = leftInterpolated{k}(i,j+1);
                end
                rightTrajectory(i,j+1) = mean(rightTrajVector);         % Mean value for joint j at percentage point i
                rightStDev(i,j+1) = std(rightTrajVector);               % Standard Deviation for joint j at percentage point i
                leftTrajectory(i,j+1) = mean(leftTrajVector);           
                leftStDev(i,j+1) = std(leftTrajVector);
                
            end
        end
        
        clear resultantData headerLine
        
        %% Kinetics (Joint Moments and Powers)
        
        dataFilename = char([TrialName 'IDResults.sto']);
        [resultantData{1}, headerLine(1,:)] = RetrieveMOTData(dataFilename); % Pull Kinetic Results from "TrialIDResults.sto"
        
        leftSuffix = '_l_moment'; 
        rightSuffix = '_r_moment';
        
        RCol = 1; % Include Time Column
        LCol = 1; % Include Time Column
        
        for i = 1:length(forPlotting)
            RCol = [RCol find(strcmp(headerLine,[forPlotting{i} rightSuffix]))];% Identify Right Side Columns of interest
            LCol = [LCol find(strcmp(headerLine,[forPlotting{i} leftSuffix]))]; % Identify Right Side Columns of interest
        end
        
        % Filter Joint Angle Curves and Populate Data Cells for Analysis
        for i = 1:length(RHS_time)-1
            startFrame = find(resultantData{1}(:,1)==RHS_time(i));
            endFrame = find(resultantData{1}(:,1)==RHS_time(i+1));
            rightKineticData{i} = normalisingFactor.*filtfilt(double(B),double(A),resultantData{1}(startFrame:endFrame,RCol)); % Filter and Normalise Moment
            rightKineticData{i}(:,1) = resultantData{1}(startFrame:endFrame,1);     % Re-place Time
            rightStancePower{i} = rightKineticData{i}.*deg2rad(rightStanceVel{i});  % Calculate Power using normalised Moment and Angular Velocity
        end
        
        for i = 1:length(LHS_time)-1
            startFrame = find(resultantData{1}(:,1)==LHS_time(i));
            endFrame = find(resultantData{1}(:,1)==LHS_time(i+1));
            leftKineticData{i} = normalisingFactor.*filtfilt(double(B),double(A),resultantData{1}(startFrame:endFrame,LCol)); % Filter and Normalise Moment
            leftKineticData{i}(:,1) = resultantData{1}(startFrame:endFrame,1);      % Re-place Time
            leftStancePower{i} = leftKineticData{i}.*deg2rad(leftStanceVel{i});     % Calculate Power using normalised Moment and Angular Velocity
        end
        
        
        % Interpolate
        % (If you want to change interpolation number, do it here.)
        
        for i = 1:length(RHS_time)-1
            rightInterpolatedMoment{i}(:,1) = linspace(rightKineticData{i}(1,1),rightKineticData{i}(end,1),interpolationPointNumber);
            rightInterpolatedPower{i}(:,1) = linspace(rightKineticData{i}(1,1),rightKineticData{i}(end,1),interpolationPointNumber);
            for j = 1:length(forPlotting)
                rightInterpolatedMoment{i}(:,j+1) = interp1(rightKineticData{i}(:,1),rightKineticData{i}(:,(1+j)),rightInterpolatedMoment{i}(:,1), 'spline');
                rightInterpolatedPower{i}(:,j+1) = interp1(rightKineticData{i}(:,1),rightStancePower{i}(:,(1+j)),rightInterpolatedMoment{i}(:,1), 'spline');
            end
        end
        
        
        for i = 1:length(LHS_time)-1
            leftInterpolatedMoment{i}(:,1) = linspace(leftKineticData{i}(1,1),leftKineticData{i}(end,1),interpolationPointNumber);
            leftInterpolatedPower{i}(:,1) = linspace(leftKineticData{i}(1,1),leftKineticData{i}(end,1),interpolationPointNumber);
            for j = 1:length(forPlotting)
                leftInterpolatedMoment{i}(:,j+1) = interp1(leftKineticData{i}(:,1),leftKineticData{i}(:,(1+j)),leftInterpolatedMoment{i}(:,1), 'spline');
                leftInterpolatedPower{i}(:,j+1) = interp1(leftKineticData{i}(:,1),leftStancePower{i}(:,(1+j)),leftInterpolatedMoment{i}(:,1), 'spline');
            end
        end
        
        % Package Mean Curves for Output (same here as for angles above)
        
        percentageValues = 0:1:100; % These also must reflect your interpolationPointNumber
        
        % Percentage x-values, preallocate
        leftTrajectoryMoment(:,1) = percentageValues;
        rightTrajectoryMoment(:,1) = percentageValues;
        leftStDevMoment(:,1) = percentageValues;
        rightStDevMoment(:,1) = percentageValues;
        rightTrajVector = zeros(length(RHS_time)-1,1);
        leftTrajVector = zeros(length(LHS_time)-1,1);
        
        % Get mean values and standard deviations
        for i = 1:length(percentageValues)
            for j = 1:length(forPlotting)
                for k = 1:(length(RHS_time)-1)
                    rightTrajVector(k) = rightInterpolatedMoment{k}(i,j+1);
                end
                for k = 1:(length(LHS_time)-1)
                    leftTrajVector(k) = leftInterpolatedMoment{k}(i,j+1);
                end
                leftTrajectoryMoment(i,j+1) = mean(leftTrajVector);
                leftStDevMoment(i,j+1) = std(leftTrajVector);
                rightTrajectoryMoment(i,j+1) = mean(rightTrajVector);
                rightStDevMoment(i,j+1) = std(rightTrajVector);
            end
        end
        
        
        % Percentage x-values, preallocate
        leftTrajectoryPower(:,1) = percentageValues;
        rightTrajectoryPower(:,1) = percentageValues;
        leftStDevPower(:,1) = percentageValues;
        rightStDevPower(:,1) = percentageValues;
        rightTrajVector = zeros(length(RHS_time)-1,1);
        leftTrajVector = zeros(length(LHS_time)-1,1);
        
        
        % Get mean values and standard deviations
        for i = 1:length(percentageValues)
            for j = 1:length(forPlotting)
                for k = 1:(length(RHS_time)-1)
                    rightTrajVector(k) = rightInterpolatedPower{k}(i,j+1);
                end
                for k = 1:(length(LHS_time)-1)
                    leftTrajVector(k) = leftInterpolatedPower{k}(i,j+1);
                end
                leftTrajectoryPower(i,j+1) = mean(leftTrajVector);
                leftStDevPower(i,j+1) = std(leftTrajVector);
                rightTrajectoryPower(i,j+1) = mean(rightTrajVector);
                rightStDevPower(i,j+1) = std(rightTrajVector);
            end
        end
        
        % Last minute translation for certain condition names (probably
        % won't affect you).
        if contains(Condition,'03')
            Condition = 'threems';
        elseif contains(Condition,'06')
            Condition = 'sixms';
        elseif contains(Condition,'09')
            Condition = 'ninems';
        end
        
        %% Output Structure
        
        % Packup Average Curves
        Left = struct('L_Angle',leftTrajectory,'L_Moment',leftTrajectoryMoment,'L_Power',leftTrajectoryPower,'L_Angle_StDev',leftStDev,'L_Moment_StDev',leftStDevMoment,'L_Power_StDev',leftStDevPower);
        Right = struct('R_Angle', rightTrajectory, 'R_Moment', rightTrajectoryMoment, 'R_Power',rightTrajectoryPower,'R_Angle_StDev',rightStDev,'R_Moment_StDev',rightStDevMoment,'R_Power_StDev',rightStDevPower);
        TrialStruct = struct('Left', Left, 'Right', Right);
        ParticipantStruct.(Condition) = TrialStruct;
        
        
        
        CompiledCurves(q).Participant = ParticipantName;
        CompiledCurves(q).leftAngle.(Condition) = leftKinematicData;
        CompiledCurves(q).leftMoment.(Condition) = leftKineticData;
        CompiledCurves(q).leftPower.(Condition) = leftStancePower;
        CompiledCurves(q).rightAngle.(Condition) = rightKinematicData;
        CompiledCurves(q).rightMoment.(Condition) = rightKineticData;
        CompiledCurves(q).rightPower.(Condition) = rightStancePower;
        
        CompiledCurves(q).leftAngleInt.(Condition) = leftInterpolated;
        CompiledCurves(q).leftMomentInt.(Condition) = leftInterpolatedMoment;
        CompiledCurves(q).leftPowerInt.(Condition) = leftInterpolatedPower;
        CompiledCurves(q).rightAngleInt.(Condition) = rightInterpolated;
        CompiledCurves(q).rightMomentInt.(Condition) = rightInterpolatedMoment;
        CompiledCurves(q).rightPowerInt.(Condition) = rightInterpolatedPower;
        
        CompiledCurves(q).joints = forPlotting;
        CompiledCurves(q).LFZ.(Condition) = L_Z;
        CompiledCurves(q).RFZ.(Condition) = R_Z;
        CompiledCurves(q).lhstime.(Condition) = LHS_time;
        CompiledCurves(q).rhstime.(Condition) = RHS_time;
        CompiledCurves(q).lhsTOtime.(Condition) = LHS_TO_time;
        CompiledCurves(q).rhsTOtime.(Condition) = RHS_TO_time;
        CompiledCurves(q).meanRHSIntFrame.(Condition) = round(mean(rightTOInt));
        CompiledCurves(q).meanLHSIntFrame.(Condition) = round(mean(leftTOInt));
        CompiledCurves(q).normalisingFactor = normalisingFactor;
        CompiledCurves(q).Pp.(Condition) = pareticPropulsion;
        CompiledCurves(q).leftStepLengths.(Condition) = leftsteps;
        CompiledCurves(q).rightStepLengths.(Condition) = rightsteps;
        CompiledCurves(q).speed.(Condition) = walkingSpeed;
        CompiledCurves(q).paretic = Leg;
        
        waitbar((q/length(ParticipantList)),loadingbar)
        
        % Old plotting step (now use plotGaitParameters outside of this
        % script, its a lot cleaner)
        %         curvesInt = struct('L_Angle',leftInterpolated,'L_Moment',leftInterpolatedMoment,'L_Power',leftInterpolatedPower,'R_Angle',rightInterpolated,'R_Moment',rightInterpolatedMoment,'R_Power',rightInterpolatedPower);
        %         plot_stance_kinematics_edited(leftTrajectory,rightTrajectory,leftStDev,rightStDev,leftTrajectoryMoment,rightTrajectoryMoment,leftStDevMoment,rightStDevMoment,leftTrajectoryPower,rightTrajectoryPower,leftStDevPower,rightStDevPower,forPlotting,ParticipantName,Leg,Condition,curvesInt);
        %         figTitle = [origFolder '\Stance Curve Plots\' ParticipantName Condition '.png'];
        %         export_fig(figTitle,'-transparent','-r300');
        cd ..
        close all
    end   % End of Condition
    CompiledCurves(q).ConditionList = ConditionList;
    CompiledCurves(q).Average = ParticipantStruct;
    CompiledCurves(q).flipped = 0;
    CompiledCurves(q).averageFlipped = 0;
    cd ..
end % End of Participant
close(loadingbar)

FlipCurves;         % Flip curves in accordance with standard reporting (e.g. Ankle Moment; plantarflexion becomes positive)
flipAverageCurves;  % Same

[peaks,knees] = curvePeaks(CompiledCurves); % Pull out peak values for analysis (haven't used this in a while, but its here if you want it)

if ~testing
    cd('C:\Users\dbak576\Desktop\Duncan Bakke Modelling Files\Analysis Working Files');
    save('sscurves.mat','CompiledCurves','peaks')
    peakXLSXwriter;
end

cd(origFolder)
disp('All finished!')
