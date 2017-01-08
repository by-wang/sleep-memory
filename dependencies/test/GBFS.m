function [tst_prec, feat_ind] = GBFS(xtr,ytr,xtv,ytv,xte,yte,sparsity,lambda)

	% GBFS code, generating all trees
	
	% input:
	% xtr = n*p, ytr = n*1 (assume -1, 1 for binary) 
	% sparsity = 1*p
	% lambda, scalar
	
	% output:
	% tst_prec: test accuracy
	% feat_ind: selected feature indices

	% set params
	options.learningrate=0.01;	% learning rate
	options.depth=4;			% CART tree depth
	options.ntrees = 100;		% total number of CART trees
	options.verbose = true;		% verbose on
	options.computefeaturecosts = @(e) computefeaturecosts(lambda,sparsity,e); 	% feature cost update function
	% loss function, we have squared loss, logistic loss, squared hinge loss, you can provide your own.
	% gbrt returns ensemble e. 
	[e,l] = gbrt(xtr,@(p)logisticloss(ytr',p),options);	
	
	% evaluate accuracy
	[tst_prec,val_prec,beststep,~] = evaluate(e,options,sparsity,xtr,ytr,xtv,ytv,xte,yte);
	tst_prec = tst_prec(beststep);

	% selected features
	[~,totalfeat] = calcost(e,sparsity);
	feat_ind = (totalfeat(beststep*10,:)~=0)';