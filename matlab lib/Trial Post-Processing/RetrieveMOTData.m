function [fileData, headerLine] = RetrieveMOTData(filename)
%RetrieveStance: Pull Data vector from a results file for use in plotting.
% Duncan Bakke, 2017
% Auckland Bioengineering Institute
fileID = fopen(filename,'r');
filetext = fileread(filename);
notInData = 1;
hadErrors = 0;
numberErrors = 0;
while notInData == 1
    curLine = fgetl(fileID);
    if strncmpi(curLine,'nRows',5)
        numRows=str2num(curLine((strfind(curLine,'=')+1):end));
    elseif strncmpi(curLine,'datarows',8)
        numRows = str2num(curLine((strfind(curLine,' ')+1):end));
    elseif strncmpi(curLine,'nColumns',8)
        numCols=str2num(curLine((strfind(curLine,'=')+1):end));
    elseif strncmpi(curLine,'datacolumns',11)
        numCols = str2num(curLine((strfind(curLine,' ')+1):end));
    elseif strncmpi(curLine,'endheader',9)
        notInData = 0;
    end
end
headerLine = cell(numCols,1);
fileData=zeros(numRows,numCols);
for i = 1:numCols
    headerLine{i} = fscanf(fileID,'%s',1);
end
if contains(filename,'MuscleForceDirection') %problematic ones
    for j = 1:numRows
        for i = 1:numCols
            valueString = fscanf(fileID,'%s',1);
            if strcmp(valueString,'-1.#IND0000') && j>1
                fileData(j,i) = fileData(j-1,i);
                numberErrors = numberErrors + 1;
                hadErrors = hadErrors + 1;
            elseif strcmp(valueString,'-1.#IND0000') && j == 1
                fileData(j,i) = -0.0001;
                numberErrors = numberErrors + 1;
                hadErrors = hadErrors + 1;
            else
                fileData(j,i) = str2num(valueString);
            end
        end
    end
else
    for j = 1:numRows
        for i = 1:numCols
            fileData(j,i) = fscanf(fileID,'%f',1);
        end
    end
end
fclose(fileID);
if hadErrors > 0
    fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Had %d errors!\n',hadErrors)
end
end

