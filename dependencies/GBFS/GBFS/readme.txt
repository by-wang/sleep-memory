matlab code for Gradient Boosted Feature Selection (GBFS)

Zhixiang (Eddie) Xu, Kilian Q. Weinberger, Olivier Chapelle
"Gradient Boosted Feature Selection"
Proc. of 20th ACM SIGKDD Conf. on Knowledge Discovery and Data Mining (KDD), New York, 2014

INSTALL:
cd mex_gen
run make.m in matlab

RUN:
1. download some data
xtr, ytr: training inputs and labels, 	input format: variables, n*p; labels: n*1
xtv, ytv: validation inputs and labels,
xte, yte: testing inputs and labels,
sparsity: sparsity information				input format: 1*p
traqs:    for ranking data, training queries
valqs:	  for ranking data, validation queries
tstqs:	  for testing data, testing queries

2. run GBFS.m to learn all trees
lambda:	regularizer for sparsity

3. evaluate on testing set, evaluate.m

See example.m for more details on regular feature selection

See example_structure.m for more details on structure feature selection




