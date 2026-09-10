#!/usr/bin/env python3
"""Compare local AMOs with the global continuity bound, retaining failures by stage."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from research_depth_common import campaign_records, write_csv, write_json


def analyze(directory: Path,output: Path):
    pairs=defaultdict(dict);statuses=defaultdict(int)
    for record,payload in campaign_records(directory):
        spec=record['specification'];statuses[payload['status']]+=1
        if spec['method']!='lex-cos':raise ValueError('local continuity experiment requires staged COS')
        key=(record['instance_sha256'],spec['cardinality'],spec.get('implied','none'),spec.get('symmetry','none'),
             payload['timeout_seconds'],spec.get('stage3_incumbent_bound',False))
        mode='local' if spec.get('zero_continuity_local',False) else 'global'
        if mode in pairs[key]:raise ValueError('duplicate continuity encoding cell')
        pairs[key][mode]=(record,payload)
    rows=[];stage_rows=[]
    for key,group in sorted(pairs.items()):
        if set(group)!= {'global','local'}:raise ValueError('missing paired control')
        record,g=group['global'];_,l=group['local']
        row={'instance':record['instance'],'instance_sha256':key[0],'encoding':key[1],
             'global_status':g['status'],'local_status':l['status'],
             'global_error':g.get('error'),'local_error':l.get('error'),
             'both_optimum':g['status']==l['status']=='OPTIMUM'}
        if row['both_optimum']:
            if any(g['metrics'][k]!=l['metrics'][k] for k in ('coverage','continuity','overtime','similarity')):
                raise ValueError('local/global optimum vector mismatch')
            row['global_over_local_elapsed']=g['elapsed_seconds']/l['elapsed_seconds']
        rows.append(row)
        for i,(a,b) in enumerate(zip(g['stages'],l['stages'])):
            if a['objective']!=b['objective']:raise ValueError('stage mismatch')
            r={**row,'stage':i+1,'objective':a['objective'],'global_stage_status':a['status'],'local_stage_status':b['status']}
            if i==0 and a['status']==b['status']=='OPTIMUM' and a['optimum']!=b['optimum']:
                raise ValueError('continuity anchors disagree')
            for metric in ('variables','hard_clauses','soft_clauses','encode_seconds','solve_seconds'):
                r[f'global_{metric}']=a.get(metric);r[f'local_{metric}']=b.get(metric)
            stage_rows.append(r)
    write_csv(output/'zero_continuity_pairs.csv',rows);write_csv(output/'zero_continuity_stages.csv',stage_rows)
    paired_runtime={}
    formula_reduction={}
    for encoding in sorted({r['encoding'] for r in rows}):
        decided=[r for r in rows if r['encoding']==encoding and r['both_optimum']]
        ratios=[r['global_over_local_elapsed'] for r in decided]
        paired_runtime[encoding]={
            'pairs':len(ratios),
            'local_faster_pairs':sum(ratio>1 for ratio in ratios),
            'median_global_over_local_elapsed':statistics.median(ratios) if ratios else None,
        }
        formula_reduction[encoding]={}
        for stage in (2,3):
            reached=[r for r in stage_rows if r['encoding']==encoding and r['stage']==stage
                     and r['global_stage_status']=='OPTIMUM' and r['local_stage_status']=='OPTIMUM'
                     and r.get('global_hard_clauses') is not None and r.get('local_hard_clauses')]
            ratios=[r['global_hard_clauses']/r['local_hard_clauses'] for r in reached]
            formula_reduction[encoding][f'stage_{stage}_median_global_over_local_hard_clauses']=statistics.median(ratios) if ratios else None
    summary={'source':str(directory),'status_counts':dict(statuses),'paired_instances_encodings':len(rows),
             'both_optimum_pairs':sum(r['both_optimum'] for r in rows),
             'solver_errors':sum(count for name,count in statuses.items() if name not in {'OPTIMUM','UNSAT','UNSATISFIABLE','TIMEOUT','TIMEOUT_FEASIBLE'}),
             'paired_runtime':paired_runtime,
             'formula_reduction':formula_reduction,
             'interpretation':'Elapsed-time ratios require both complete optimal solves. Formula sizes can still diagnose the encoding when a later solver stage fails; these are not end-to-end speedups.'}
    write_json(output/'zero_continuity_analysis.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();result=analyze(a.result_dir,a.output_dir);print(result)
    raise SystemExit(2 if result['solver_errors'] else 0)
