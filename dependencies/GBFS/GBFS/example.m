% addpath to the CART tree generator, and make
addpath('mex_gen');
cd mex_gen
make
cd ..

% load in data (PCMAC data set)
load datasets/pcmac.mat

% 80-20 split to generate validation
rand('seed',1);
itr=ismember(1:length(ytr),randsample(1:length(ytr),ceil(0.8*length(ytr))));
xtv = xtr(~itr,:);
ytv = ytr(~itr);
xtr = xtr(itr,:);
ytr = ytr(itr);

% generate GBFS trees, with lambda = 1;
lambda = 1;
sparsity = ones(1,size(xtr,2)); % same sparsity for every feature (no pre-defined sparsity patterns)
[tst_prec, feat_ind] = GBFS(xtr,ytr,xtv,ytv,xte,yte,sparsity,lambda);



