#!/usr/bin/env python3
"""Paired policy-by-encoding analysis with explicit timeout censoring."""
from __future__ import annotations
import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_depth_common import campaign_records, write_csv, write_json


def analyze(directory: Path, output: Path):
    stages, contrasts = [], []
    blocks = defaultdict(dict)
    statuses = defaultdict(int)
    for record, payload in campaign_records(directory):
        spec = record['specification']
        enc = payload['cardinality_encoding']
        method = payload['method']
        statuses[payload['status']] += 1
        key = (record['instance_sha256'], spec.get('implied','none'), spec.get('symmetry','none'),
               spec.get('wc',1),spec.get('wo',1),payload['timeout_seconds'])
        cell = (method, enc)
        if cell in blocks[key]:
            raise ValueError(f'duplicate policy/encoding cell: {key}:{cell}')
        blocks[key][cell] = payload
        previous = {}
        for i,stage in enumerate(payload['stages']):
            stages.append({'run_id':record['run_id'],'instance_sha256':record['instance_sha256'],
                           'instance':record['instance'],'method':method,'encoding':enc,'stage':i+1,
                           'objective':stage['objective'],'run_status':payload['status'],
                           'stage_status':stage['status'],
                           'preceding_continuity_optimum':previous.get('continuity'),
                           'preceding_overtime_optimum':previous.get('overtime'),
                           **{k:stage.get(k) for k in ('encode_seconds','solve_seconds','variables','hard_clauses','soft_clauses')}})
            if stage['status']=='OPTIMUM': previous[stage['objective']] = stage['optimum']
    required = [(p,e) for p in ('weighted','lex-cos') for e in ('sorting-network','totalizer')]
    for key,cells in sorted(blocks.items()):
        if not all(c in cells for c in required):
            raise ValueError(f'incomplete paired block: {key}')
        row={'instance_sha256':key[0], 'implied':key[1], 'symmetry':key[2], 'wc':key[3],'wo':key[4],
             'timeout_seconds':key[5], 'all_four_optimum':all(cells[c]['status']=='OPTIMUM' for c in required)}
        for p,e in required:
            x=cells[p,e]; label=f'{p}_{e}'
            row[f'{label}_status']=x['status']
            row[f'{label}_seconds']=x['elapsed_seconds']
            row[f'{label}_par2']=2*x['timeout_seconds'] if x['status'].startswith('TIMEOUT') else x['elapsed_seconds']
        if row['all_four_optimum']:
            ratios={p: cells[p,'sorting-network']['elapsed_seconds']/cells[p,'totalizer']['elapsed_seconds'] for p in ('weighted','lex-cos')}
            row.update({'weighted_sn_over_tot':ratios['weighted'],'cos_sn_over_tot':ratios['lex-cos'],
                        'policy_encoding_log_contrast':math.log(ratios['lex-cos'])-math.log(ratios['weighted'])})
        contrasts.append(row)
    aggregates=[]
    grouped=defaultdict(list)
    for s in stages: grouped[(s['method'],s['encoding'],s['stage'],s['objective'])].append(s)
    for key,group in sorted(grouped.items()):
        row=dict(zip(('method','encoding','stage','objective'),key))
        row.update(reached_runs=len(group),optimum_stages=sum(g['stage_status']=='OPTIMUM' for g in group),
                   timeout_stages=sum(g['stage_status'].startswith('TIMEOUT') for g in group))
        for field in ('encode_seconds','solve_seconds','variables','hard_clauses','soft_clauses'):
            values=[g[field] for g in group if g[field] is not None]
            row[f'median_{field}_reached']=statistics.median(values) if values else None
        aggregates.append(row)
    exact=[r['policy_encoding_log_contrast'] for r in contrasts if r['all_four_optimum']]
    summary={'source':str(directory),'status_counts':dict(statuses),'paired_blocks':len(contrasts),
             'all_four_optimum_blocks':len(exact),'median_log_contrast':statistics.median(exact) if exact else None,
             'interpretation':'Positive log contrast means the SN/TOT runtime ratio is larger for COS than Weighted. Exact ratios use four-optimum blocks only; report selection and timeout counts. PAR2 is a timeout-penalized score, not observed solve time. Reached-stage medians are descriptive, not paired speedup estimates.'}
    write_csv(output/'stage_encoding_runs.csv',stages)
    write_csv(output/'stage_encoding_summary.csv',aggregates)
    write_csv(output/'policy_encoding_contrasts.csv',contrasts)
    write_json(output/'stage_encoding_validation.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();print(analyze(a.result_dir,a.output_dir))
