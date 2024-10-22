function prepareTrialFromVicon(model, trial, directory, inputDirectory)
% prepareTrialFromVicon: A function to condition and collate trial data and
% setup all necessary OpenSim analysis xmls.
% Inputs:   model = Name of subject, assuming model file is "Subject.osim".
%           trial = Name of motion capture trial.
%           directory = Location of output.
%           inputDirectory = Location of input files.
% NOTE: currently EMG analysis is disabled but not dysfunctional (I think!)
%       If you want to use it, change "badEMG" to 0.

%% Customisation
rotationRequired = 1; %alter if Nexus ever gets fixed
recalculateCOP = 1;
% badEMGTrials: If you have mostly fine EMG with some trials that failed.
badEMGTrials = {'None.'};
% badEMGTrials = {'SAFIST015_SS21_20Jun_ss_035ms_02','SAFIST015_SS21_20Jun_fast_075ms_02','SAFIST015_SS42_20Jun_ss_035ms_01','SAFIST015_SS42_20Jun_fast_055ms_01','SAFIST015_SS52_ss_04ms_02','SAFIST015_SS52_fast_07ms_01','SS77_SAFIST015_18Jun_fast_04ms_02','SAFIST015_19Jun_SS90_ss_035ms_01','SAFIST015_19Jun_SS90_fast_055ms_01','_12Mar_ss_12ms_01'};
badEMG = 1; %REALLOW EMG HERE IF NEEDED
if any(contains(badEMGTrials,trial))
    badEMG = 1;
end

% FROM JULIE
% zMat = [0 0 1; 0 1 0; -1 0 0];
% yMat = [-1 0 0; 0 1 0; 0 0 -1];
% fullMat = zMat*yMat;
theta = 90;
fullMat = [cosd(theta) 0 sind(theta); 0 1 0; -sind(theta) 0 cosd(theta)]; % Rotation because Nexus has some funny axis ideas (note that this ONLY affects pelvis rotations)

%% Filename Generation
%Identify files from Vicon Export to read
trcFilename = fullfile(inputDirectory,strcat(trial,'.trc'));
motFilename = fullfile(inputDirectory,strcat(trial,'.mot'));
emgFilename = fullfile(inputDirectory,strcat(trial,'_EMG.mot'));


% Check if files are present (from EITHER naming convention)
if ~isfile(trcFilename)
    trcFilename = fullfile(inputDirectory,strcat(model,trial,'.trc'));
end

if ~isfile(trcFilename)
    error('Marker TRC file required!')
end

if ~isfile(motFilename)
    motFilename = fullfile(inputDirectory,strcat(model,trial,'.mot'));
end

if ~isfile(motFilename)
    error('GRF MOT file required!')
end

if ~isfile(emgFilename)
    fprintf('No EMG for subject %s.\n',model)
    badEMG = 1;
end

% If you have set up gait events in Vicon Nexus, then export them via
% "Export ASCII" as .txt and import as below (decomment related commands
% further on)
% txtfilename = fullfile(inputDirectory,strcat(trial,'.txt'));

mkdir(directory,model);
mkdir(strcat(directory,'\',model),trial);
newFolder = fullfile(directory,model,trial);
IKfilename = 'IKSetup.xml';
IDfilename = 'IDSetup.xml';
exLoadsFilename = 'ExternalLoads.xml';
MuscleAnalysisfilename = 'MuscleAnalysisSetup.xml';
MuscleForceDirectionfilename = 'MuscleForceDirectionSetup.xml';

%% Pull in Exported Vicon Files, Identify time range of interest

% (Note: This approach differs with regard to available event data.)
[frames, markerNames, markers, datarate] = readTRC(trcFilename);
% [left, right] = readEventsTXT(txtfilename);
disp(model);
disp(trial);
if ~badEMG
    [EMGheaders,EMGdata,EMGfreq] = readEMGMOT(emgFilename);
end
[GRFheaders,fullGRFdata] = readMOT(motFilename);

% If you have trial data saved seperately, use this point to insert steps
% and plates as below.
% trialData = ?
% steps = trialData.steps;
% plates = trialData.plates;

if contains(model,'SS') || contains(model,'AB') %If SS or AB, recorded at Millenium. (If you used Millenium Gait Lab, use these values!)
    steps = {'l','r'};
    plates = [1,2];
end

steps = {'l','r'};
plates = [1,2];

% If you're modelling overground walking (i.e. not a treadmill) then steps
% and plates need to be ALL steps and plates used!

% If overground, this ensures correct time limits.
% if strcmp(steps{1},'L')
%     timerange(1) = min(right.off);
%     timerange(2) = max(left.strike);
% else
%     timerange(1) = min(left.off);
%     timerange(2) = max(right.strike);
% end
%


% Alter here if you want different time ranges. 0.020 is to allow for EMG
% propogation delay.
timerange(1) = round(max(frames(1,2),0)+0.020,3);
timerange(2) = frames(end,2);

indexStart = find(frames(:,2) == timerange(1));
indexEnd = find(frames(:,2) == timerange(2));

framerange(1) = frames(indexStart,1);
framerange(2) = frames(indexEnd,1);


%% IK File
[trimmedMarkers, trimmedFrames] = trimTRC(markers,frames,[indexStart indexEnd]);                % Trim according to your time of interest.
[goodMarkers,goodMarkerNames,badMarkerNames] = removeBadMarkers(trimmedMarkers,markerNames);    % Get rid of anything missing for 10 frames 

% NOTE: This should never happen since you've already filled gaps in Nexus!
% This function will also get rid of spurious marker traces.


if rotationRequired % Rotate the marker trajectories to make the axes line up if necessary.
    for i = 1:length(goodMarkers)
        markerMat = [goodMarkers(i).x, goodMarkers(i).y, goodMarkers(i).z];
        transformedMarkerMat = markerMat*fullMat;
        goodMarkers(i).x = transformedMarkerMat(:,1);
        goodMarkers(i).y = transformedMarkerMat(:,2);
        goodMarkers(i).z = transformedMarkerMat(:,3);
    end
end

writeMarkersToTRC(fullfile(newFolder,strcat(trial,'.trc')),goodMarkers,goodMarkerNames,datarate,trimmedFrames(:,1),trimmedFrames(:,2),'mm'); % Write new TRC file, ready for OpenSim.

%Note that this function manually edits the xml file. This is better done
%using the OpenSim APIs if you want to.
% ikTime = changeIKXMLFile(IKfilename,trial,timerange,goodMarkerNames,badMarkerNames,model,directory); %#ok<NASGU>
% fullIKfilename = strcat(trial,IKfilename);
% xmlShorten(fullIKfilename);
% % Move File
% movefile(fullIKfilename, newFolder);

% There now ought to be an IKSetup xml file in place with consistent file
% references (as well as altered marker weightings)

%% ID Files

%Define rate of GRF capture
GRFrate = (length(fullGRFdata)-1)/(fullGRFdata(end,1)-fullGRFdata(1,1));

% Get Original Vertical Forces
originalFys = fullGRFdata(:,find(not(~contains(GRFheaders, 'vy'))));

% Condition GRFData (filter)
[b, a] = butter(4, (10/(GRFrate/2)));
newGRFdata(:,1) = fullGRFdata(:,1);
for i = 2:length(GRFheaders)
    newGRFdata(:,i) = filtfilt(b, a, fullGRFdata(:,i));
end

% Re-Zero GRFs
filterPlate = reZeroFilter(originalFys); % Make sure anywhere that was zero in original recording is zero in the trial version.

for i = 2:length(GRFheaders)
    if isempty(strfind(GRFheaders{i},'p')) && i<=(length(GRFheaders)/2) %this relies on centre of pressure columns containing the letter 'p' and NO OTHER COLUMNS doing so.
        newGRFdata(:,i) = filterPlate(:,1).*newGRFdata(:,i);
    elseif isempty(strfind(GRFheaders{i},'p'))
        newGRFdata(:,i) = filterPlate(:,2).*newGRFdata(:,i);
    end
end

if recalculateCOP
    % Define for recaluclating CoP
    xoffset = [0.2385 0.7275]; % This is hardcoded by necessity; change this if you didn't record at Millenium
    yoffset = [0 0];
    vzInds = find(contains(GRFheaders,'vy')); % OpenSIM COORDS
    pxInds = find(contains(GRFheaders,'px'));
    pyInds = find(contains(GRFheaders,'pz'));
    
    % Back Calculate Moment Measurements
    for i = 1:length(plates)
        sideInds = find(contains(GRFheaders,num2str(i)));
        fZ(i,:) = fullGRFdata(:,intersect(vzInds,sideInds));
        pX(i,:) = fullGRFdata(:,intersect(pxInds,sideInds));
        pY(i,:) = fullGRFdata(:,intersect(pyInds,sideInds));
        oldmY(i,:) = (xoffset(i)-pX(i,:)).*fZ(i,:);
        oldmX(i,:) = (yoffset(i)+pY(i,:)).*fZ(i,:);
        % Check out kwon3d for explanations on these formulae
    end
    
    for i = 1:length(plates)
        mY(i,:) = filtfilt(b,a,oldmY(i,:));
        mX(i,:) = filtfilt(b,a,oldmX(i,:));
    end
    
    for i = 1:length(plates)
        mY(i,:) = filterPlate(:,i)'.*mY(i,:);
        mX(i,:) = filterPlate(:,i)'.*mX(i,:);
    end
    
    % Recalculate CoP with Filtered forces and moments
    for i = 1:length(plates)
        sideInds = find(contains(GRFheaders,num2str(i)));
        newfZ = newGRFdata(:,intersect(vzInds,sideInds));
        newpX(i,:) = xoffset(i)-(mY(i,:)./newfZ');
        newpY(i,:) = yoffset(i)+(mX(i,:)./newfZ');
        
        for j=1:length(newpX(i,:))
            if isnan(newpX(i,j))
                newpX(i,j)=0;
                newpY(i,j)=0;
            end
        end
        
        newGRFdata(:,intersect(pxInds,sideInds)) = newpX(i,:);
        newGRFdata(:,intersect(pyInds,sideInds)) = newpY(i,:);
        
%         Plotter for new CoP if validation is needed.
        figure;
        plot(pX(i,:),pY(i,:),'*');
        hold on
        plot(newpX(i,:),newpY(i,:),'x');
%         fprintf('Just associated new CoP for plate %i\n',i)
    end
    
end

GRFdata = newGRFdata(find(newGRFdata(:,1)==timerange(1)):find(newGRFdata(:,1)==timerange(2)),:);

%newHeaders = fixGRFheaders(GRFh;eaders,steps,plates); % To ensure it works with OpenSim
newHeaders = GRFheaders;
if rotationRequired % Have to rotate GRFs too of course.
    numGRFsets = floor(length(newHeaders)/3);
    for i = 1:numGRFsets
        %avoid time column
        inds = (((i-1)*3)+2):(((i)*3)+1);
        curMat = [GRFdata(:,inds(1)), GRFdata(:,inds(2)), GRFdata(:,inds(3))];
        transformedGRFMat = curMat*fullMat;
        GRFdata(:,inds(1)) = transformedGRFMat(:,1);
        GRFdata(:,inds(2)) = transformedGRFMat(:,2);
        GRFdata(:,inds(3)) = transformedGRFMat(:,3);
    end
end

writeMOT(fullfile(newFolder,strcat(trial,'.mot')),newHeaders,GRFdata); %Write up a new GRF MOT file too.
return
IDerr = changeIDXMLFile(IDfilename,trial,timerange,model,directory,10); %#ok<NASGU>
xmlShorten(strcat(trial,IDfilename));

ExLerr = changeLoadXMLFile(exLoadsFilename,trial,model,directory,10); %#ok<NASGU>
xmlShorten(strcat(trial,exLoadsFilename));

% % Move Files
fullIDfilename = strcat(trial,IDfilename);
movefile(fullIDfilename, newFolder);
fullexLoadsFilename = strcat(trial,exLoadsFilename);
movefile(fullexLoadsFilename, newFolder);

% There now ought to be IDSetup xml and External Loads xml files in place
% with consistent file references.

%% EMG Processing
if ~badEMG
    EMGenv = envelopeEMG(EMGdata,EMGfreq);
    if strfind(EMGheaders{1},'Frame') && strfind(EMGheaders{2},'Frame')
        EMGenv(:,1:2) = EMGdata(:,1:2);
    end
    emgDelay = 0.02; % 2 frames (@100 Hz) or 4 frames (@200 Hz)
    
    frameOffset = emgDelay/(1/datarate);
    
    EMGstart = find(EMGenv(:,1)==framerange(1)-frameOffset);
    EMGend = find(EMGenv(:,1)==framerange(2)-frameOffset);
    EMGtime = [(timerange(1)-emgDelay):(1/EMGfreq):(timerange(2)-emgDelay)]';
    
    clippedEMG = EMGenv(EMGstart:EMGend,:);
    EMG = [EMGtime clippedEMG(:,3:end)];
    EMGlabels = {'time' EMGheaders{3:end}};
    
    writeEMG(EMG,EMGlabels,fullfile(newFolder,strcat(trial,'_EMG.mot')));
end

%% Muscle Analysis Files

% These are only necessary if you are planning on doing muscle force
% analysis!
MAerr = changeMuscleAnalysisXMLFile(MuscleAnalysisfilename,trial,timerange,model);
xmlShorten(strcat(trial,MuscleAnalysisfilename));

MAerr = changeMuscleForceDirectionXMLFile(MuscleForceDirectionfilename,trial,timerange,model);
xmlShorten(strcat(trial,MuscleForceDirectionfilename));

% Move Files
fullMALoadsFilename = strcat(trial,MuscleAnalysisfilename);
movefile(fullMALoadsFilename, newFolder);
fullMFDLoadsFilename = strcat(trial,MuscleForceDirectionfilename);
movefile(fullMFDLoadsFilename, newFolder);
end