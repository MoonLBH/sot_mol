import argparse, json
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--task',default='Ranolazine_MPO'); args=ap.parse_args()
    df=pd.read_csv(args.csv)
    out={}
    out['official_guacamol_frac']=float(df.get('official_guacamol', pd.Series([0]*len(df))).mean()) if 'official_guacamol' in df.columns else None
    for col,name in [('raw_sim_ranolazine_AP','raw_sim'),('component_sim_ranolazine_AP','comp_sim'),('raw_num_F','raw_numF'),('component_num_F','comp_numF')]:
        if col in df.columns:
            s=df[col].dropna(); out[name]={'mean':float(s.mean()),'q90':float(s.quantile(0.9)),'top10':float(s.nlargest(min(10,len(s))).mean()),'max':float(s.max())}
    if 'score' in df.columns and 'component_sim_ranolazine_AP' in df.columns:
        out['corr_score_comp_sim']=float(df['score'].corr(df['component_sim_ranolazine_AP']))
    if 'score' in df.columns and 'component_num_F' in df.columns:
        out['corr_score_comp_numF']=float(df['score'].corr(df['component_num_F']))
    if 'top_selected' in df.columns:
        out['top_frac_numF_gt0']=float((df[df['top_selected']==True]['raw_num_F']>0).mean()) if 'raw_num_F' in df.columns else None
    if 'bottom_reason' in df.columns:
        out['bottom_reason_dist']=df['bottom_reason'].value_counts(normalize=True).to_dict()
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
