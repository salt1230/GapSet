"""Paired whole-gesture-group bootstrap conditional on observed users/seeds."""
import numpy as np

def paired_intervals(counts,group_users,model_names,condition_names,*,replicates=501,seed=20261020):
    """counts[M,S,C,G,6,6]; all receivers/windows are summed inside each G.

    The same sampled group weights apply to every model, seed and condition.
    Users and training seeds remain fixed. No population-level p-values.
    """
    a=np.asarray(counts);users=np.asarray(group_users)
    if a.ndim!=6 or a.shape[-2:]!=(6,6) or a.shape[3]!=len(users):raise ValueError('Expected grouped six-class counts')
    if not np.issubdtype(a.dtype,np.integer) or (a<0).any():raise ValueError('Nonnegative integer counts required')
    if a.shape[0]!=len(model_names) or a.shape[2]!=len(condition_names):raise ValueError('Axis labels differ')
    if not np.all(a.sum((-2,-1))==a[0,0,0].sum((-2,-1))[None,None,None]):raise ValueError('Group sample counts must stay paired')
    if replicates<2:raise ValueError('At least two replicates')
    weights=np.zeros((replicates,len(users)),dtype=np.float64);rng=np.random.default_rng(seed)
    for user in sorted(set(users.tolist())):
        ids=np.flatnonzero(users==user)
        for b in range(replicates):weights[b,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
    def macro(cm):
        den=cm.sum(-1)+cm.sum(-2)
        return np.divide(2*np.diagonal(cm,axis1=-2,axis2=-1),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(-1)*100
    observed=macro(a.sum(3));boot=np.empty((a.shape[0],a.shape[1],replicates,a.shape[2]))
    for m in range(a.shape[0]):
        for s in range(a.shape[1]):
            matrix=a[m,s].transpose(1,0,2,3).reshape(len(users),-1).astype(np.float64)
            boot[m,s]=macro((weights@matrix).reshape(replicates,a.shape[2],6,6))
    ref=model_names.index('ours_gapset_relative_mass_control');clean=condition_names.index('clean')
    challenge=[i for i,c in enumerate(condition_names) if c!='clean'];rows=[]
    for m,name in enumerate(model_names):
        if m==ref:continue
        for label,indices in [('clean_macro_f1',[clean]),('challenge45_mean_macro_f1',challenge)]:
            if not indices:continue
            point=(observed[ref][:,indices].mean(1)-observed[m][:,indices].mean(1)).mean()
            paired=(boot[ref][:,:,indices].mean(2)-boot[m][:,:,indices].mean(2)).mean(0)
            rows.append(dict(comparator=name,metric=label,mean_seed_difference_pp=float(point),
                conditional_group_percentile95=np.quantile(paired,[.025,.975],method='linear').tolist()))
    return dict(comparisons=rows,replicates=replicates,rng_seed=seed,
      conditional_on_observed_users_and_training_seeds=True,no_p_values=True,
      not_independent_test_or_population_significance=True)
