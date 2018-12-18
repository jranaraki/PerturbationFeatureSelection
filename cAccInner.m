% Programmed by Javad Rahimipour Anaraki on 29/05/18
% Ph.D. Candidate
% Department of Computer Science
% Memorial University of Newfoundland
% jra066 [AT] mun [DOT] ca | www.cs.mun.ca/~jra066

% Input: Selected features
% Output: Classification accuracy using different classifier

function clsOut = cAccInner(selF)

%%================================Data=====================================
global data;
[r, ~] = size(data);
trainSize = floor(.7 * r);
train = data(1:trainSize, selF);
test = data(trainSize+1:end, selF);
nClass = length(unique(data(:, end)));

%%==============================Classifiers================================
Model = fitctree(train(:, 1:end-1), train(:, end));
%Model = fitcecoc(train(:, 1:end-1), train(:, end));
%Model = fitcknn(train(:, 1:end-1), train(:, end));

predicted = predict(Model, test(:, 1:end-1));

%%========================Classification Accuracy==========================
C = confusionmat(test(:, end), predicted);
term = diag(C) ./ sum(C, 2);
clsOut = 1/nClass * sum(term) * 100;

return
