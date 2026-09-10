#!/usr/bin/env python3
"""Join reverified policy differences to matched capacity variants and parent families."""
from __future__ import annotations
import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_policy_structure import analyze as structure
from research_depth_common import resolve_instance, write_csv, write_json


def analyze(directory: Path, manifest: Path, output: Path):
    generated=json.loads(manifest.read_text());by_hash={}
    for row in generated['instances']:
        if row['instance_sha256'] in by_hash:raise ValueError('duplicate variant hash')
        resolve_instance(row['instance'],row['instance_sha256'])
        by_hash[row['instance_sha256']]=row
    verified=structure(directory,output/'structure')
    with (output/'structure/policy_structure_pairs.csv').open(newline='') as stream: pairs=list(csv.DictReader(stream))
    enriched=[];groups=defaultdict(list)
    for pair in pairs:
        source=by_hash[pair['instance_sha256']]
        match=re.search(r'instance_u(\d+)_a\d+_v\d+_seed(\d+)_',source['parent_instance'])
        if not match:raise ValueError('unknown parent family naming')
        row={**pair,'parent_sha256':source['parent_sha256'],'parent_instance':source['parent_instance'],
             'family':f'u{match[1]}_seed{match[2]}','target_rho':source['rho'],'realized_rho':source['realized_rho'],
             'normal_fraction':source['normal_fraction'],'anchor':source['anchor']}
        for k in ('continuity','overtime','similarity'):
            row[f'{k}_right_minus_left']=int(row[f'{k}_right'])-int(row[f'{k}_left'])
        row['different_objective_vector']=any(row[f'{k}_right_minus_left'] for k in ('continuity','overtime','similarity'))
        enriched.append(row)
        groups[(row['target_rho'],row['normal_fraction'],row['backend'],row['left'],row['right'])].append(row)
    summary=[]
    for key,rows in sorted(groups.items()):
        row=dict(zip(('target_rho','normal_fraction','backend','left','right'),key))
        row.update(paired_instances=len(rows),parent_families=len({r['family'] for r in rows}),
                   conflicts=sum(r['different_objective_vector'] for r in rows))
        for k in ('continuity','overtime','similarity'):
            row[f'mean_{k}_right_minus_left']=statistics.mean(r[f'{k}_right_minus_left'] for r in rows)
        summary.append(row)
    write_csv(output/'load_policy_pairs.csv',enriched);write_csv(output/'load_cell_summary.csv',summary)
    result={'verified':verified,'cells':len(summary),'paired_rows':len(enriched),
            'interpretation':'Descriptive matched comparisons. Nested variants and capacity cells are not independent samples; use parent family as the resampling unit for uncertainty estimates. These variants preserve feasibility and do not test infeasibility transitions.'}
    write_json(output/'load_sweep_analysis.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();print(analyze(a.result_dir,a.manifest,a.output_dir))
