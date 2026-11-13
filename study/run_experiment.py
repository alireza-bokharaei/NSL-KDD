"""NSL-KDD independent reproducibility experiment. Python 3.13; sklearn 1.6.1."""
import os, json, hashlib, platform, time, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import pandas as pd,numpy as np,sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from pathlib import Path
base=Path(__file__).parent
train=pd.read_csv(base/'data/KDDTrain.txt',header=None); test=pd.read_csv(base/'data/KDDTest.txt',header=None)
assert train.shape==(125973,43) and test.shape==(22544,43)
assert not train.isna().any().any() and not test.isna().any().any()
X=train.iloc[:,:41];y=(train.iloc[:,41]!='normal').astype(int)
Xt=test.iloc[:,:41];yt=(test.iloc[:,41]!='normal').astype(int)
itr,ival=train_test_split(np.arange(len(train)),test_size=0.2,random_state=42,stratify=y)
cat=[1,2,3];num=[c for c in range(41) if c not in cat]
# log1p applied only to nonnegative numeric features; validate all data beforehand.
assert (X[num].min().min()>=0) and (Xt[num].min().min()>=0)
def build(kind):
 numpipe=Pipeline([('log1p',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),('scale',StandardScaler())])
 prep=ColumnTransformer([('num',numpipe,num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)],sparse_threshold=0.3)
 model=LogisticRegression(max_iter=1000,random_state=42,solver='lbfgs') if kind=='Logistic regression' else RandomForestClassifier(n_estimators=80,min_samples_leaf=2,random_state=42,n_jobs=2)
 return Pipeline([('preprocess',prep),('classifier',model)])
def results(model,x,y,labels,seen):
 probs=model.predict_proba(x)[:,1];pred=(probs>=0.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
 old=(labels!='normal') & labels.isin(seen);new=(labels!='normal') & ~labels.isin(seen)
 def part(mask):
  ix=np.asarray(mask);n=int(ix.sum());return {'n':n,'detected':int(pred[ix].sum()),'recall':float(pred[ix].mean()) if n else None}
 return dict(n=len(y),attacks=int(y.sum()),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred)),recall=float(recall_score(y,pred)),f1=float(f1_score(y,pred)),fpr=float(fp/(fp+tn)),roc_auc=float(roc_auc_score(y,probs)),average_precision=float(average_precision_score(y,probs)),seen_attacks=part(old),unseen_attacks=part(new))
out={'dataset':{'train_rows':len(train),'test_rows':len(test),'train_normal':int((y==0).sum()),'train_attack':int(y.sum()),'test_normal':int((yt==0).sum()),'test_attack':int(yt.sum()),'unseen_test_attack_labels':sorted(set(test.iloc[:,41])-set(train.iloc[:,41]))},'config':{'seed':42,'threshold':0.5,'holdout':'80/20 stratified on KDDTrain+; no test-set model tuning','formal':'train full KDDTrain+, evaluate full KDDTest+','features':'41 predictors; exclude label and difficulty','hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (base/'data').glob('*.txt')},'python':platform.python_version(),'scikit-learn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},'models':{}}
for name in ['Logistic regression','Random forest']:
 print('starting',name,flush=True); start=time.time();m=build(name);m.fit(X.iloc[itr],y.iloc[itr]); hold=results(m,X.iloc[ival],y.iloc[ival],train.iloc[ival,41],set(train.iloc[itr,41]));print('holdout',hold,flush=True)
 m=build(name);m.fit(X,y);official=results(m,Xt,yt,test.iloc[:,41],set(train.iloc[:,41])); print('official',official,flush=True)
 out['models'][name]={'random_holdout':hold,'official_test':official,'seconds':round(time.time()-start,2)}
features=list(range(41))
match=test.merge(train[features].drop_duplicates(),on=features,how='inner')
train2=train.copy(); test2=test.copy();train2[43]=(train2[41]!='normal');test2[43]=(test2[41]!='normal')
matching_labels=test2.merge(train2[features+[43]].drop_duplicates(),on=features+[43],how='inner')
out['dataset']['test_rows_exact_41_feature_overlap']=int(len(match))
out['dataset']['test_rows_overlap_same_binary_label']=int(len(matching_labels))
(base/'results.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('DONE',flush=True)
