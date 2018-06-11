% Unset the arff from Matlab environment

arffpath = '~/so/Matlab/arff';

% Add arff to the Matlab path
rmpath(arffpath);

cdir = pwd;
cd(arffpath);
javarmpath('weka.jar');
cd(cdir);
