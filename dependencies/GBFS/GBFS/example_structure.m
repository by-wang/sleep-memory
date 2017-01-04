% addpath to the CART tree generator, and make
clear all
clear global

addpath('mex_gen');
cd mex_gen
make
cd ..

% load in data (Colon data set)
load datasets/colon.mat
% idx indicates which features belong to which group.

% random split training and testing
itr=ismember(1:length(y),randsample(1:length(y),ceil(0.8*length(y))));
ytr = y(itr);
yte = y(~itr);
xtr = x(itr,:);
xte = x(~itr,:);

% generate GBFS trees, with lambda = 1;
lambda = 1;
global sparsity;
sparsity = ones(1,size(xtr,2)); % same sparsity for every feature at the beginning
global groupsparsity;
groupsparsity = ones(length(unique(idx)),1); % feature group, in Colon, there are 9 groups

[tst_prec, feat_ind, groups] = GBFS_structure(xtr,ytr,xte,yte,sparsity,idx,lambda);



