function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    predictions = (pval < epsilon);
    res = [predictions yval];

    matrix = [1 1;1 0;0 1;0 0];
    matrix_res = zeros(4,1);

    for i = 1:size(matrix,1)
        temp = res == matrix(i,:);
        matrix_res(i,:) = size(find((sum(temp,2) == 2) == 1),1);
    end

    
    if matrix_res(1)==0
        recall = 0;
        prescision = 0;
    else
        recall = matrix_res(1)/(matrix_res(1)+ matrix_res(3));
        prescision = matrix_res(1)/(matrix_res(1)+matrix_res(2));
    end

    F1 = 2*prescision*recall/(prescision + recall);






    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
