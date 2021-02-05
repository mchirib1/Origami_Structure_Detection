%% MATLAB_ConfusionMatrix
%% 
%  Calculates confusion matrix from CNN predictions and manually classified 
%  structures. 
%
%% Steps
%
%  0. Input number of images for confusion matrix.
%  1. Import predictions and manual classification results.
%  2. Match manual classifications to predictions.
%  3. Tabulate and export results.
%
%% 0. Input number of images for confusion matrix
prompt = {'Number of images to combine in confusion matrix'};
dlg_title = 'Input';
num_lines = 1;
defaultans = {'1'};
%
Out = inputdlg(prompt,dlg_title,num_lines,defaultans);
Z = str2double(Out(1));
%
Results = zeros(2,2); % Creating confusion matrix (TP,FP;FN,TN)
%
filepathPred = uigetdir(pwd,'Select folder containing predictions list');
filepathAnnot = uigetdir(pwd,'Select folder containing annotations list');
%
for j = 1:Z
%% 1. Import predictions and manual classification results.
[name, ~]=uigetfile('*.txt','Select predictions list',filepathPred) ;
%
Predict = importdata(fullfile(filepathPred,name)) ;
Manual = importdata(fullfile(filepathAnnot,name)) ;
%
%% 2. Match manual classifications to predictions.
ResultsTemp = zeros(2,2);
%
M = size(Manual,1);
N = size(Predict,1);
%
for i = 1:M
    for k = 1:N
        if ((Manual(i,2)<=Predict(k,2)+0.5*Predict(k,4))&&...
                (Manual(i,2)>=Predict(k,2)-0.5*Predict(k,4))&&...
                (Manual(i,3)<=Predict(k,3)+0.5*Predict(k,5))&&...
                (Manual(i,3)>=Predict(k,3)-0.5*Predict(k,5)))
        Manual(i,2:5)=0; %Remove structure from manual list
        Predict(k,2:5)=0; % Remove BB from prediction list
        ResultsTemp(1,1)=ResultsTemp(1,1)+1;
        break
        else % defines case where no common structure identified
            if k == N
               ResultsTemp(2,1)=ResultsTemp(2,1)+1;
            end
        end
    end
end
ResultsTemp(1,2) = N - (ResultsTemp(1,1) + ResultsTemp(1,2)); % left over
Results = Results + ResultsTemp; % Adds new results to cumulative results
clear i k m n Image Manual Predict Width Height nameImage nameManual namePredict
end
csvwrite(fullfile(pwd,'ConfusionMatrix.csv'),Results)
 