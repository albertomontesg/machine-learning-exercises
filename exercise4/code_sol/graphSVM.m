% DESCRIPTION: Function to visualize 2-class SVMs 
% applied to bivariate data. 

% INPUT:
% model: an object created by fitcsvm function with fitted SVM model
% X: Data matrix. Must have only 2 columns (bivariate data)
% class: vector with as many elements as there are rows in X. Numeric class
% labels of data points. 

% OUTPUT:
% None. The function only graphs the model. 

function [] = graphSVM(model, X, class)

%identify the class labels
classes = unique(class);

%identify which rows correspond to class 1 and class2
indx1 = find(class==classes(1));
indx2 = find(class==classes(2));

%identify support vectors
svInd = model.IsSupportVector;

%Create a grid of possible bivariate predictor values. We will use this
%grid to add "score" contour lines and background colour which indicates
%the two different class prediction regions
h = 0.02; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)),min(X(:,2)):h:max(X(:,2)));
xGrid = [X1(:),X2(:)];
%calculate score for each value on grid
[label,score] = predict(model,xGrid);
scoreGrid = reshape(score(:,1),size(X1,1),size(X2,2));
[~, maxScore] = max(score, [],2);

%create plot
figure
%plot green and purple background indicating the class prediction regions
p1=gscatter(xGrid(:,1),xGrid(:,2),maxScore,[0.1 0.5 0.5; 0.5 0.1 0.5]);
hold on
%Add scatter of training samples, blue for class 1, red for class 2
p2=plot(X(indx1,1),X(indx1,2),'bx',X(indx2,1),X(indx2,2),'rx');
hold on
%Indicate which samples are support vectors
p3=plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10);
%Add contour lines of "score" (distance to the hyperplane in enlarged 
%feature space)
contour(X1,X2,scoreGrid)
% Of particuar interest is contour line which corresponds to score=0,
% ie the separating hyperplane projected onto the original feature space.
contour(X1,X2,scoreGrid, [0,0], 'black')
colorbar;
legend HIDE
hold off

end

