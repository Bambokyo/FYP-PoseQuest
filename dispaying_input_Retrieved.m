% testing
addpath(genpath('D:\University\FYP\Implementation\HDM05-Parser\HDM05-Parser\parser'))
addpath(genpath('D:\University\FYP\Implementation\HDM05-Parser\HDM05-Parser\parser\ASFAMCparser'))
addpath(genpath('D:\University\FYP\Implementation\HDM05-Parser\HDM05-Parser\animate'))
addpath('D:\University\FYP\Implementation\HDM05-Parser\HDM05-Parser\quaternions')

% displaying the data
[skel,mot] = readMocap('HDM_bd.asf', 'X.amc');
%% animate asf/amc file
animate(skel, mot);
disp('Retireived');

% displaying the data
[skel2,mot2] = readMocap('HDM_bd.asf', 'Y.amc');
%% animate asf/amc file
animate(skel2, mot2);
disp('input pose');