%
% Convert .amc file to csv, which contains the first PCA component 
% of the z-normalized motion capture data.
%
% Subject 49 trial #3: jump up and down, hop on one foot
% Data downloaded from: http://mocap.cs.cmu.edu/subjects/49/49_03.amc
%

% amc_to_matrix available from: http://graphics.cs.cmu.edu/software/amc_to_matrix.m
D = amc_to_matrix('56_03.amc');
%D = amc_to_matrix('49_03.amc');
D = zscore(D);
[C, X] = pca(D);
T = (1:length(X))';
plot(T, X(:, 1));
csvwrite('../data/mocap2.csv', [T, X(:, 1)]);
