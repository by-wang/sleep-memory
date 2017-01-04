function [tst_prec, feat_ind, groups] = GBFS_structure(xtr,ytr,xte,yte,sparsity,idx,lambda)

	% GBFS code, generating all trees
	
	% input:
	% xtr = n*p, ytr = n*1 (assume -1, 1 for binary) 
	% sparsity = 1*p
	% lambda, scalar
	
	% output:
	% tst_prec: test accuracy
	% feat_ind: selected feature indices
	% groups: how many groups are selected
	
	% set params
	options.ntrees = 50;
	options.depth = 3;
	options.verbose = true;
	options.learningrate=0.001;	% learning rate
	if lambda ~= 0	
		options.computefeaturecosts = @(e) groupfeaturecost(lambda,sparsity,idx,e); 	% feature sparsity update function
	end
	[e,l] = gbrt_hack(xtr,@(p)logisticloss(ytr',p),options);
	
	% evaluate accuracy
	eall = cell2mat(e{2}');
	tst_preds = evalensemble_c(xte,eall,options.depth,0.001,10);
	tst_preds = cumsum(tst_preds,2);
	tst_preds = (sign(tst_preds)+1)/2;
	tst_prec = bsxfun(@eq, tst_preds, yte);
	tst_prec = mean(tst_prec,1);
	tst_prec = tst_prec(end);

	% selected features
	[~,totalfeat] = calcost(e,sparsity);
	feat_ind = (totalfeat(end,:)~=0)';
	global groupsparsity;
	groups = length(groupsparsity) - sum(groupsparsity);