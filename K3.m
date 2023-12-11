addpath(genpath('D:\University\FYP\Implementation\HDM_01-01_amc\vLSH-master'))

% Loop through different values of L
L_values = [2, 4, 8, 16, 32];

% Initialize a flag to determine if it's the first iteration
first_iteration = true;

% Load the .mat files
queryData = load('queryDataset.mat');  % Load the query poses
datasetData = load('wholeDataset.mat');  % Load the whole dataset poses

% Transpose the data
queryPoses = queryData.queryDataset.pos;  % Transpose the query data
datasetPoses = datasetData.wholeDataset.pos;  % Transpose the whole dataset data

% Take the first 500 rows from queryPoses
queryPoses = queryPoses;

% Take the first 1000 rows from datasetPoses
datasetPoses = datasetPoses;

% Initialize variables to store evaluation metrics
mpjseList = [];
pckList = [];

% In the context of Locality-Sensitive Hashing (LSH),
% the parameter M represents the number of dimensions
% in the projection space. When data points are hashed
% in LSH, they are first projected onto a lower-dimensional
% space (projection space) before being assigned to buckets.
% The value of M determines the dimensionality of this projection
% space. A higher value of M generally results in a more accurate 
% hash, but it also increases the computational cost. On the other
% hand, a lower value of M might reduce computational cost
% but could potentially decrease the accuracy of the hash.
% The choice of the optimal value for M depends on various 
% factors such as the characteristics of your dataset,
% the desired trade-off between accuracy and efficiency,
% and the specific requirements of your application.

% Loop through different values of L
for L = L_values
    % Construct LSH index tables
    M = 10;  % # of dimensions at projection space (adjust based on your needs)
    W = 1000; % bucket width
    lshStruct = lshConstruct(datasetPoses, L, M, W);

     % Perform LSH search for the query matrix
    K = 32;  % Choose an appropriate value for K
    start_time = tic;
    [idsLSH, ~, ~] = lshSearch(queryPoses, datasetPoses, lshStruct, K);
    retrieval_time = toc(start_time);

    % Ensure indices are within bounds
    validIndices = idsLSH <= size(datasetPoses, 1);

    % Calculate evaluation metrics for each valid query pose
    for i = 1:size(queryPoses, 1)
        % Get the valid neighbors from the LSH search
        validNeighbors = datasetPoses(idsLSH(i, validIndices(i, :)), :);
        
        disp(['sze of valid veinborurs ): ', num2str(size(validNeighbors))]);
        disp(['query size  ): ', num2str(size(queryPoses(i,:)))]);

        % Calculate Mean Per Joint Squared Error (MPJSE)
        mpjse = mean(sqrt(sum((queryPoses(i, :) - validNeighbors).^2, 2)));
        mpjseList = [mpjseList; mpjse];

        % Compute PCK (Percentage of Correct Keypoints)
        threshold = 30; % Set a threshold for correctness
        correctKeypoints = sqrt(sum((queryPoses(i, :) - validNeighbors).^2, 2)) < threshold;
        pck = sum(correctKeypoints) / length(correctKeypoints);
        pckList = [pckList; pck];
    end

    % Calculate the mean of each evaluation metric
    meanMpjse = mean(mpjseList);
    meanPck = mean(pckList);

    % Display results
    disp(['Number of Tables (L): ', num2str(L)]);
    disp(['Mean Per Joint Squared Error (MPJSE): ', num2str(meanMpjse)]);
    disp(['Percentage of Correct Keypoints (PCK): ', num2str(meanPck)]);
    disp(['Retrieval Time: ', num2str(retrieval_time), ' seconds']);

    % Save data to CSV file
    if first_iteration
        header = {'L_value', 'MPJSE', 'PCK', 'Retrieval_Time'};
    end
    data = {L, meanMpjse, meanPck, retrieval_time};

    % Check if the file already exists to write the header only once
    if first_iteration
        writecell(header, 'results3.csv', 'WriteMode', 'overwrite');
        first_iteration = false;
    end

    % Append data to the CSV file
    writecell(data, 'results3.csv', 'WriteMode', 'append');

    % Clear variables for the next iteration
    mpjseList = [];
    pckList = [];
end
