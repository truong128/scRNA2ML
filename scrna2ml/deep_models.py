"""scikit-learn wrapper for the tabular neural networks with donor-safe fixed-epoch training.
No internal random early-stopping split, no outcome-informed batch correction.
"""
import numpy as np
import torch
from torch import nn
from sklearn.base import ClassifierMixin,BaseEstimator
from sklearn.utils.validation import check_X_y,check_array,check_is_fitted
from .tabular_architectures import ARCH_CATALOG,build_model
class TorchTabularClassifier(ClassifierMixin,BaseEstimator):
 def __init__(self,architecture='ft_transformer',epochs=30,random_state=42):
  self.architecture=architecture;self.epochs=epochs;self.random_state=random_state
 def fit(self,X,y):
  X,y=check_X_y(X,y,dtype=np.float32);self.classes_,encoded=np.unique(y,return_inverse=True);self.n_features_in_=X.shape[1]
  torch.set_num_threads(1);torch.manual_seed(self.random_state)
  cfg=ARCH_CATALOG[self.architecture];self.network_=build_model(cfg['model_type'],[],X.shape[1],cfg)
  # Original networks have binary heads; preserve feature extractor and adapt output to K classes.
  if isinstance(self.network_.head,nn.Sequential):
   last=self.network_.head[-1];self.network_.head[-1]=nn.Linear(last.in_features,len(self.classes_))
  else:self.network_.head=nn.Linear(self.network_.head.in_features,len(self.classes_))
  cont=torch.from_numpy(X);cat=torch.empty((len(X),0),dtype=torch.long);target=torch.from_numpy(encoded.astype(np.int64))
  weights=len(y)/(len(self.classes_)*np.bincount(encoded));loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
  optimizer=torch.optim.AdamW(self.network_.parameters(),lr=cfg['lr'],weight_decay=.01)
  self.network_.train()
  for _ in range(self.epochs):
   optimizer.zero_grad();loss=loss_fn(self.network_(cat,cont),target)
   if not torch.isfinite(loss):raise ValueError('Nonfinite neural-network loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(self.network_.parameters(),1.0);optimizer.step()
  self.network_.eval();return self
 def predict_proba(self,X):
  check_is_fitted(self,'network_');X=check_array(X,dtype=np.float32)
  if X.shape[1]!=self.n_features_in_:raise ValueError('Feature count mismatch')
  with torch.no_grad():p=torch.softmax(self.network_(torch.empty((len(X),0),dtype=torch.long),torch.from_numpy(X)),dim=1).numpy()
  return p
 def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
