"""Reproducible grouped comparisons, followed by Bayesian tuning of the CV winner."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from scipy.special import softmax, logit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from .core import FEATURES

SEED = 41
MODELS = {
    'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300)),
    'naive_bayes':GaussianNB(),
    'decision_tree':DecisionTreeClassifier(min_samples_leaf=32,max_depth=12,random_state=SEED),
    'extra_trees':ExtraTreesClassifier(n_estimators=32,min_samples_leaf=8,max_depth=18,n_jobs=2,random_state=SEED),
    'boosting':HistGradientBoostingClassifier(max_iter=100,early_stopping=False,random_state=SEED)
}
# One influential parameter per family; continuous log scale or integer grid.
SEARCH = {'boosting':('learning_rate',np.geomspace(.025,.3,33)),
          'logistic':('logisticregression__C',np.geomspace(.01,100,33)),
          'naive_bayes':('var_smoothing',np.geomspace(1e-12,1e-3,33)),
          'extra_trees':('min_samples_leaf',np.arange(2,35)),
          'decision_tree':('min_samples_leaf',np.arange(4,100,3))}
def probabilities(model,x):
    shape=x.shape[:2]
    pos=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(shape)
    return softmax(logit(np.clip(pos,1e-7,1-1e-7)),axis=1)

def load(split):
    out=[]
    for p in sorted(split.glob('*.npz')):
        with np.load(p) as data:
            out.append(dict(name=p.stem, **{k:data[k] for k in ('x','truth','edges','ids')}))
    if not out: raise ValueError(f'No prepared trips in {split}')
    return out

def training_rows(trips, cap):
    rng=np.random.default_rng(SEED)
    blocks=[]
    targets=[]
    for trip in trips:
        y=trip['ids']==trip['truth'][:,None]
        rows=np.flatnonzero(y.any(axis=1))
        if len(rows)>cap: rows=rng.choice(rows,cap,replace=False)
        blocks.append(trip['x'][rows].reshape(-1,len(FEATURES)))
        targets.append(y[rows].ravel())
    return np.concatenate(blocks),np.concatenate(targets)

def evaluation(model,trips):
    rows=[]
    for trip in trips:
        x,ids,truth=trip['x'],trip['ids'],trip['truth']
        start=perf_counter()
        p=probabilities(model,x) if model is not None else softmax(-np.expm1(x[:,:,0].astype(float)),axis=1)
        best=p.argmax(axis=1)
        pred=ids[np.arange(len(ids)),best]
        seconds=perf_counter()-start
        actual=ids==truth[:,None]
        covered=actual.any(axis=1)
        prob=(p*actual).sum(axis=1)[covered]
        edges=trip['edges'][np.arange(len(ids)),best]
        # Strict adjacent arc continuity; skips across intermediate arcs count as breaks.
        connected=(pred[1:]==pred[:-1])|(edges[:-1,1]==edges[1:,0])
        pr,re,f1,_=precision_recall_fscore_support(truth,pred,average='macro',zero_division=0)
        rows.append(dict(trip=trip['name'],n=len(ids),top1=float(np.mean(pred==truth)),
            precision=pr,recall=re,macro_f1=f1,covered=int(covered.sum()),
            log_loss=float(-np.log(np.maximum(prob,1e-300)).mean()) if len(prob) else None,
            continuity=float(connected.mean()) if len(connected) else None,score_seconds=seconds))
    return rows

def summary(rows):
    return {k:float(np.mean([r[k] for r in rows if r[k] is not None]))
            for k in ('top1','macro_f1','log_loss','continuity','score_seconds','precision','recall')}

def fit_evaluate(template,fit_data,trips):
    begin=perf_counter()
    model=clone(template).fit(*fit_data)
    elapsed=perf_counter()-begin
    return model,evaluation(model,trips),elapsed

def save_csv(name,rows):
    with open(Path('results')/name,'w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

def run(cache,cap,trials):
    train,test=load(cache/'train'),load(cache/'student_test')
    splits=GroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(train,groups=[t['name'] for t in train])
    folds=[]; manifest=[]
    for fold,(fit,val) in enumerate(splits):
        folds.append((training_rows([train[i] for i in fit],cap),[train[i] for i in val]))
        manifest.extend(dict(trip=train[i]['name'],fold=fold) for i in val)
    save_csv('folds.csv',manifest)
    all_fit=training_rows(train,cap)
    results=[]; details=[]
    def compare(name,model):
        cv=[]
        for i,(fit,valid) in enumerate(folds):
            rows=evaluation(None,valid) if model is None else fit_evaluate(model,fit,valid)[1]
            cv.extend(rows)
            details.extend(dict(model=name,partition='cv',fold=i,**r) for r in rows)
        if model is None: fitted=None;elapsed=0;rows=evaluation(None,test)
        else: fitted,rows,elapsed=fit_evaluate(model,all_fit,test)
        details.extend(dict(model=name,partition='student_test',fold='',**r) for r in rows)
        fit_metrics=summary(evaluation(fitted,train))
        result=dict(model=name,train_top1=fit_metrics['top1'],train_macro_f1=fit_metrics['macro_f1'],cv_top1=summary(cv)['top1'],cv_macro_f1=summary(cv)['macro_f1'],
                    train_seconds=elapsed,**summary(rows))
        results.append(result)
        print(name,round(result['cv_top1'],4),round(result['top1'],4),flush=True)
        save_csv('model_comparison.csv',results);save_csv('trip_metrics.csv',details)
        return fitted
    compare('nearest',None)
    for name,model in MODELS.items(): compare(name,model)
    winner=max(results[1:],key=lambda r:r['cv_top1'])['model']
    key,grid=SEARCH[winner]
    x=np.linspace(0,1,len(grid))[:,None]
    chosen=[];scores=[];history=[]
    for step in range(trials):
        if step<3: index=[0,len(grid)//2,len(grid)-1][step]
        else:
            gp=GaussianProcessRegressor(kernel=Matern(length_scale=.25,nu=2.5),normalize_y=True,
                                        alpha=1e-5,optimizer=None).fit(x[chosen],scores)
            mu,std=gp.predict(x,return_std=True)
            gain=mu-max(scores); z=gain/np.maximum(std,1e-12)
            ei=gain*norm.cdf(z)+std*norm.pdf(z);ei[chosen]=-np.inf
            index=int(ei.argmax())
        value=grid[index].item()
        model=clone(MODELS[winner]).set_params(**{key:value})
        start=perf_counter()
        rows=[r for fit,valid in folds for r in fit_evaluate(model,fit,valid)[1]]
        chosen.append(index);scores.append(summary(rows)['top1'])
        history.append(dict(trial=step+1,family=winner,parameter=key,value=value,
            cv_top1=scores[-1],cv_macro_f1=summary(rows)['macro_f1'],seconds=perf_counter()-start))
        save_csv('bayesian_trials.csv',history)
    best=max(history,key=lambda r:r['cv_top1'])
    baseline=next(r for r in results if r['model']==winner)
    settings={key:best['value']} if best['cv_top1']>=baseline['cv_top1'] else {}
    tuned=clone(MODELS[winner]).set_params(**{key:best['value']})
    fitted=compare(winner+'_tuned',tuned)
    if not settings: fitted=clone(MODELS[winner]).fit(*all_fit)
    k=train[0]['ids'].shape[1]
    joblib.dump(dict(model=fitted,features=FEATURES,k=k), 'artifacts/model.joblib',compress=3)
    config=dict(family=winner,selected=settings,candidate_count=k,seed=SEED,
        observation_cap_per_fit_trip=cap,search_trials=trials,search_space=grid.tolist(),
        objective='mean per-trip observation Top-1',models={n:str(m) for n,m in MODELS.items()})
    Path('results/config.json').write_text(json.dumps(config,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=Path('.cache/features'))
    p.add_argument('--cap',type=int,default=256)
    p.add_argument('--trials',type=int,default=6,choices=range(4,33))
    args=p.parse_args()
    run(args.cache,args.cap,args.trials)
