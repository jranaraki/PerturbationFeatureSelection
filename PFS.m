% Programmed by Javad Rahimipour Anaraki on 29/05/18
% Ph.D. Candidate
% Department of Computer Science
% Memorial University of Newfoundland
% jra066 [AT] mun [DOT] ca | www.cs.mun.ca/~jra066

% Input: A dataset
% Output: Selected feautres and the resulting classification accuracy using PFS

warning off;
global data;
%%================================Data=====================================
strArray{1} = 'LSVT_voice_rehabilitation.csv';
strArray{2} = 'madelon.csv';
strArray{3} = 'colon.csv';
strArray{4} = 'lung.csv';
strArray{5} = 'lymphoma.csv';
strArray{6} = 'GLIOMA.csv';
strArray{7} = 'leukemia.csv';
strArray{8} = 'ALLAML.csv';

path = ['Data/'];
%%=============================Parameters==================================
runIter = 10;
cu = 100000;
cl = 1000000;
clMethod = 0; %0 for kMeans and 1 for cMeans

%%================================Main=====================================
for n = [1:8]
    tic
    disp(['Loading ', strArray{n}, ' ...']);
    
    data = readLargeCSV(strcat(path, strArray{n}));
    
    %Impute missing values
    if(sum(isnan(data(:))) > 0)
        data = knnimpute(data);
    end
    
    [r, ~] = size(data);
    
    %==========================Data Prepration=============================
    if (data(end,1) == r)
        data = data(:, 2:end);
    end
    
    [r, c] = size(data);
    allF = c - 1;
    clusters = min(min([allF, r]), rank(data(:,1:end-1)));
    out = cell(runIter * clusters, 4);
    eliteCluster = zeros(runIter, 4);
    
    fprintf("Run ");
    for run = 1:runIter
        fprintf("%d,", run);

	%==============================Shuffling===========================
        data = data(randperm(size(data, 1)), :);

        %==============================Variables===========================
        A = data(:,1:end-1);
        B = data(:,end);
        deg = zeros(allF, 1);
        degToB = zeros(allF, 1);
        
        %============================Normalization=========================
        A = normc(A);
        
        %========================Perturbantion matrix======================
        iA = pinv(A);
        X = iA * B;
        nB = norm(B);
        baseAngle = real(acosd(sum((A * X) .* B) / (norm(A * X) * nB)));
        minPer = min(A)/cl;
        maxPer = max(A)/cu;
        
        perVal = (maxPer - minPer) .* rand(r,1) + minPer;
        pA = A + perVal;
        piA = pinv(pA);
        pX = abs(piA * B - X);
        
        %=============================Experiments==========================
        for j = 1:allF
            tmpA = A;
            tmpX = X;
            tmpA(:, j) = [];
            tmpX(j, :) = [];
            tmpB = tmpA * tmpX;
            ntmpB = norm(tmpB);
            deg(j) = real(abs(acosd(dot(tmpB, B) / (ntmpB * nB)) - baseAngle));
            degToB(j) = real(abs(acosd(dot(A(:, j), B) / (norm(A(:, j)) * nB))));
        end
        
        %==============================Ranking=============================
        deg = normc(deg);
        pX = normc(pX);
        degToB = normc(degToB);
        tmpRank = [(1:allF)', deg];
        
        for f = 1:allF
            tmpRank(f, 3) = mean(degToB(tmpRank(1:f, 1)));
            tmpRank(f, 4) = pX(tmpRank(f, 1));
        end
        
        %============================Clustering============================
        idx = zeros(allF, 2);
        
        for cluster=2:clusters
            centroid = zeros(cluster, 1);
            
            if(~clMethod)
                [idx(:, 2), centers] = kmeans(tmpRank(:, 2:4), cluster);
            else
                options = [2.0 100 1e-5 0];
                [centers, mF] = fcm(tmpRank(:, 2:4), cluster,options);
                [~, idx(:, 2)] = max(mF', [], 2);
            end
            
            tmp = 1:allF;
            idx(:, 1) = tmp';
            
            for cl=1:cluster
                [~, tmpIdx] = min(sum((tmpRank(idx(:, 2) == cl, 2:end) - centers(cl, :)).^2, 2));
                
                if(isempty(tmpIdx))
                    centroid(cl) = 0;
                    continue;
                end
                
                idxs = tmpRank(idx(:, 2) == cl, 1);
                centroid(cl) = idxs(tmpIdx);
            end
            
            centroid = centroid((centroid > 0));
            
            out{(run - 1) * clusters + cluster, 1} = centroid';
            out{(run - 1) * clusters + cluster, 2} = cAccInner([centroid', c]);
            out{(run - 1) * clusters + cluster, 3} = out{(run - 1) * clusters + cluster, 2} / length(centroid');
            
            if (out{(run - 1) * clusters + cluster, 2} > eliteCluster(run, 3))
                eliteCluster(run, 1) = (run - 1) * clusters + cluster;
                eliteCluster(run, 2) = length(out{(run - 1) * clusters + cluster, 1});
                eliteCluster(run, 3) = cAccOutter([out{(run - 1) * clusters + cluster, 1}, c]);
            end
        end
    end
    
    emptyIndex = cellfun(@isempty,out);
    out(emptyIndex) = {0};
    
    eliteCluster(:, 4) = eliteCluster(:, 3) ./ eliteCluster(:, 2);
    [~, bestMeasure] = max(eliteCluster(:, 4));
    SFMeasure = out{eliteCluster(bestMeasure, 1)};
    CAMeasure = eliteCluster(bestMeasure, 3);
    
    [~, bestAccuracy] = max(eliteCluster(:, 3));
    SFAccuracy = out{eliteCluster(bestAccuracy, 1)};
    CAAccuracy = eliteCluster(bestAccuracy, 3);
    
    meanACC = mean(eliteCluster(:, 3));
    meanSelF = mean(eliteCluster(:, 2));
    originalACC = cAccOutter([1:c]);
    
    fprintf("\b\n");
    
    disp('  ---------------Selection criterion: Best Measure---------------  ');
    disp(['  |SF| = ', num2str(length(SFMeasure)), ', CA = ', num2str(CAMeasure), '%, Mean(|SF|) = ', num2str(meanSelF), ', Mean(CA) = ', num2str(meanACC), '%, Measure = ', num2str(CAMeasure/length(SFMeasure)), ', CA(original) = ', num2str(originalACC), '%']);
    disp(['  Selected subset = [', num2str(SFMeasure), ']']);
    
    disp('  ---------------Selection criterion: Best Accuracy---------------  ');
    disp(['  |SF| = ', num2str(length(SFAccuracy)), ', CA = ', num2str(CAAccuracy), '%, Mean(|SF|) = ', num2str(meanSelF), ', Mean(CA) = ', num2str(meanACC), '%, Measure = ', num2str(CAAccuracy/length(SFAccuracy)), ', CA(original) = ', num2str(originalACC), '%']);
    disp(['  Selected subset = [', num2str(SFAccuracy), ']']);
    fprintf("\n");
    
    toc
    fprintf("\n");
end
