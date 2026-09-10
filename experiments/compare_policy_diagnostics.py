#!/usr/bin/env python3
"""Audit independently certified diagnostic objectives across two solver campaigns."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_policy_diagnostics import analyze
from research_depth_common import campaign_records, write_csv, write_json


def index(directory):
    result={}
    for record,payload in campaign_records(directory):
        s=record['specification']
        key=(record['instance_sha256'],s['method'],s.get('wc',1),s.get('wo',1),
             s.get('continuity_slack'),s.get('probe_objective'),s.get('probe_sense'))
        if key in result:raise ValueError('duplicate audit cell')
        result[key]=payload
    return result


def signature(payload):
    m=payload['metrics'];method=payload['method']
    if method=='weighted':return (m['coverage'],m['weighted_reference_score'])
    if method=='weighted-face':return (payload['weighted_optimum_reference'],m[payload['probe_objective']])
    if method=='continuity-budget':return (payload['continuity_optimum_reference'],m['overtime'],m['similarity'])
    return tuple(m[k] for k in ('coverage','continuity','overtime','similarity'))


def compare(left: Path,right: Path,output: Path):
    analyze(left,output/'left');analyze(right,output/'right')
    a,b=index(left),index(right)
    if set(a)!=set(b):raise ValueError('audit requires identical requested cells')
    rows=[]
    for key in sorted(a,key=str):
        x,y=a[key],b[key]
        if x['backend']==y['backend']:raise ValueError('independent backend audit requires different backends')
        both=x['status']==y['status']=='OPTIMUM'
        rows.append({'instance_sha256':key[0],'method':key[1],'wc':key[2],'wo':key[3],
                     'continuity_slack':key[4],'probe_objective':key[5],'probe_sense':key[6],
                     'left_backend':x['backend'],'right_backend':y['backend'],
                     'left_status':x['status'],'right_status':y['status'],
                     'both_optimum':both,'objective_agreement':signature(x)==signature(y) if both else None})
    write_csv(output/'diagnostic_backend_audit.csv',rows)
    result={'requested_pairs':len(rows),'both_optimum_pairs':sum(r['both_optimum'] for r in rows),
            'disagreements':sum(r['objective_agreement'] is False for r in rows)}
    result['complete_agreement']=result['both_optimum_pairs']==len(rows) and result['disagreements']==0
    write_json(output/'diagnostic_backend_audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();result=compare(a.left,a.right,a.output_dir);print(result)
    raise SystemExit(0 if result['complete_agreement'] else 2)
