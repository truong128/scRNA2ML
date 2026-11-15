"""Registry of candidate classifiers (36 with all optional dependencies) plus a prior baseline, sized for small donor cohorts; no global resampling."""
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,Perceptron,PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier

def registry():
 lr=lambda **kw:LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42,**kw)
 tree=lambda:DecisionTreeClassifier(max_depth=3,min_samples_leaf=2,class_weight='balanced',random_state=42)
 rf=lambda:RandomForestClassifier(n_estimators=150,min_samples_leaf=2,class_weight='balanced',n_jobs=1,random_state=42)
 models={
 'logistic_regression':lr(), 'multinom_reg':lr(solver='saga'),
 'svm_rbf':SVC(class_weight='balanced'), 'svm_linear':SVC(kernel='linear',class_weight='balanced'),
 'decision_tree':tree(), 'random_forest':rf(),
 'gradient_boosting':HistGradientBoostingClassifier(max_iter=100,min_samples_leaf=2,max_leaf_nodes=5,random_state=42),
 'extra_trees':ExtraTreesClassifier(n_estimators=150,min_samples_leaf=2,class_weight='balanced',n_jobs=1,random_state=42),
 'adaboost':AdaBoostClassifier(n_estimators=60,algorithm='SAMME',random_state=42),
 'bag_tree':BaggingClassifier(estimator=tree(),n_estimators=25,max_samples=.8,random_state=42,n_jobs=1),
 'bagging':BaggingClassifier(estimator=tree(),n_estimators=100,random_state=42,n_jobs=1),
 'knn':KNeighborsClassifier(n_neighbors=1,weights='distance'),
 'lda':LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),
 'qda':QuadraticDiscriminantAnalysis(reg_param=.9), 'naive_bayes':GaussianNB(),
 'mlp':MLPClassifier(hidden_layer_sizes=(16,),alpha=1,max_iter=500,early_stopping=False,random_state=42),
 'logistic_lasso':lr(penalty='l1',solver='liblinear'),
 'logistic_elastic_net':lr(penalty='elasticnet',solver='saga',l1_ratio=.5),
 'ridge':RidgeClassifier(class_weight='balanced'),
 'sgd_logistic':SGDClassifier(loss='log_loss',alpha=.01,class_weight='balanced',max_iter=2000,random_state=42),
 'perceptron':Perceptron(class_weight='balanced',random_state=42),
 'passive_aggressive':PassiveAggressiveClassifier(class_weight='balanced',random_state=42),
 'svm_poly':SVC(kernel='poly',degree=2,class_weight='balanced'),
 'classic_gradient_boosting':GradientBoostingClassifier(n_estimators=60,max_depth=1,random_state=42),
 'nearest_centroid':NearestCentroid(), 'bernoulli_nb':BernoulliNB(),
 'gaussian_process':GaussianProcessClassifier(random_state=42),
 'voting_soft':VotingClassifier([('lr',lr()),('rf',rf()),('nb',GaussianNB())],voting='soft'),
 'voting_hard':VotingClassifier([('lr',lr()),('rf',rf()),('svm',SVC())],voting='hard'),
 'dummy_prior':DummyClassifier(strategy='prior')}
 unavailable={}
 for name,module,cls,kw in [
 ('xgboost','xgboost','XGBClassifier',dict(n_estimators=100,max_depth=2,learning_rate=.05,n_jobs=1,random_state=42)),
 ('lightgbm','lightgbm','LGBMClassifier',dict(n_estimators=100,num_leaves=4,min_child_samples=2,verbosity=-1,n_jobs=1,random_state=42)),
 ('catboost','catboost','CatBoostClassifier',dict(iterations=100,depth=3,verbose=False,thread_count=1,random_seed=42,allow_writing_files=False))]:
  try:
   import importlib
   models[name]=getattr(importlib.import_module(module),cls)(**kw)
  except ImportError:unavailable[name]='Dependency not installed'
 try:
  from .deep_models import TorchTabularClassifier
  for name in ['ft_transformer','ft_transformer_large','tabular_resnet','realmlp']:
   models[name]=TorchTabularClassifier(architecture=name)
 except ImportError as e:
  unavailable['deep_architectures']=str(e)
 return models,unavailable
