#!/usr/bin/env python3
"""Analyze certified Weighted optimal-face bounds and continuity-budget curves."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from research_depth_common import ROOT, campaign_records, resolve_instance, write_csv, write_json
sys.path.insert(0,str(ROOT/'src/proposed'))
from hcorap.io import read_instance
from hcorap.model import Assignment
from hcorap.metrics import verify_assignments


def analyze(directory: Path, output: Path):
    blocks=defaultdict(dict);status=defaultdict(int)
    for record,payload in campaign_records(directory):
        status[payload['status']]+=1
        path=resolve_instance(record['instance'],record['instance_sha256'])
        if payload['metrics'] is not None:
            if 'assignments' not in payload: raise ValueError('diagnostics require saved assignments')
            checked=verify_assignments(read_instance(path),tuple(Assignment(*a) for a in payload['assignments']))
            if not checked.valid:raise ValueError(checked.violations)
            m=checked.metrics
            if (m.coverage,m.continuity_penalty,m.overtime,m.similarity)!=tuple(payload['metrics'][k] for k in ('coverage','continuity','overtime','similarity')):
                raise ValueError('independent metric mismatch')
        spec=record['specification']; key=(record['instance_sha256'],payload['backend'],spec.get('wc',1),spec.get('wo',1))
        cell=(payload['method'],spec.get('probe_objective') if payload['method']=='weighted-face' else None,
              spec.get('probe_sense') if payload['method']=='weighted-face' else None,
              spec.get('continuity_slack',0) if payload['method']=='continuity-budget' else None)
        if cell in blocks[key]:raise ValueError(f'duplicate diagnostic cell: {key}:{cell}')
        blocks[key][cell]=(record,payload)
    faces=[];curves=[];checks=[]
    for key,cells in sorted(blocks.items()):
        first=next(iter(cells.values()))[0]
        base=dict(zip(('instance_sha256','backend','wc','wo'),key));base['instance']=first['instance']
        get=lambda cell:cells.get(cell,({},{}))[1]
        cos=get(('lex-cos',None,None,None));weighted=get(('weighted',None,None,None))
        face={**base};refs=set(); complete=True
        for objective in ('continuity','overtime'):
            for sense in ('min','max'):
                p=get(('weighted-face',objective,sense,None));label=f'{objective}_{sense}'
                face[f'{label}_status']=p.get('status','MISSING')
                if p.get('status')=='OPTIMUM':
                    ref=p['weighted_optimum_reference'];refs.add(ref)
                    if ref!=p['metrics']['weighted_reference_score']:raise ValueError('weighted face score mismatch')
                    stages=p['stages']
                    if len(stages)!=2 or any(s['status']!='OPTIMUM' for s in stages) or stages[0]['incumbent']!=ref:
                        raise ValueError('uncertified weighted anchor/probe')
                    face[label]=p['metrics'][objective]
                else:complete=False
        if len(refs)>1:raise ValueError('inconsistent Weighted optima within one block')
        if refs and weighted.get('status')=='OPTIMUM' and weighted['metrics']['weighted_reference_score'] not in refs:
            raise ValueError('Weighted anchor disagrees with diagnostic probes')
        face['weighted_optimum']=next(iter(refs),None);face['all_four_probes_optimum']=complete
        if complete:
            for obj in ('continuity','overtime'):
                if face[f'{obj}_min']>face[f'{obj}_max']:raise ValueError('inverted optimal-face interval')
            if cos.get('status')=='OPTIMUM':
                face['cos_continuity']=cos['metrics']['continuity']
                face['cos_overtime']=cos['metrics']['overtime']
                face['all_weighted_optima_have_worse_continuity']=face['continuity_min']>face['cos_continuity']
                face['all_weighted_optima_have_worse_overtime']=face['overtime_min']>face['cos_overtime']
                face['all_weighted_optima_have_worse_both']=face['all_weighted_optima_have_worse_continuity'] and face['all_weighted_optima_have_worse_overtime']
            face['weighted_face_has_continuity_variation']=face['continuity_min']<face['continuity_max']
            face['weighted_face_has_overtime_variation']=face['overtime_min']<face['overtime_max']
        faces.append(face)
        for cell,(record,p) in sorted(cells.items(),key=lambda item:str(item[0])):
            if cell[0]!='continuity-budget':continue
            row={**base,'slack':cell[3],'status':p['status'],'continuity_reference':p.get('continuity_optimum_reference')}
            if p['metrics']:
                row.update({k:p['metrics'][k] for k in ('continuity','overtime','similarity')})
            if p['status']=='OPTIMUM':
                ref=p['continuity_optimum_reference']
                if row['continuity']>ref+row['slack']:raise ValueError('continuity budget violated')
                if len(p['stages'])!=3 or any(s['status']!='OPTIMUM' for s in p['stages']) or p['stages'][0]['incumbent']!=ref:
                    raise ValueError('uncertified continuity budget curve')
                if cos.get('status')=='OPTIMUM' and ref!=cos['metrics']['continuity']:raise ValueError('continuity anchor mismatch')
                if row['slack']==0 and cos.get('status')=='OPTIMUM' and any(row[k]!=cos['metrics'][k] for k in ('continuity','overtime','similarity')):
                    raise ValueError('k=0 does not reproduce COS vector')
            curves.append(row)
        certified=sorted([r for r in curves if all(r[k]==base[k] for k in base) and r['status']=='OPTIMUM'],key=lambda r:r['slack'])
        for left,right in zip(certified,certified[1:]):
            if right['continuity_reference']!=left['continuity_reference'] or right['overtime']>left['overtime'] or (right['overtime']==left['overtime'] and right['similarity']<left['similarity']):
                raise ValueError('non-monotone certified budget curve')
            checks.append({**base,'slack_from':left['slack'],'slack_to':right['slack'],
                           'overtime_reduction':left['overtime']-right['overtime'],
                           'similarity_change':right['similarity']-left['similarity']})
    write_csv(output/'weighted_face_intervals.csv',faces);write_csv(output/'continuity_budget_curves.csv',curves)
    write_csv(output/'continuity_budget_changes.csv',checks)
    budget_overtime_instances={r['instance_sha256'] for r in checks if r['overtime_reduction']>0}
    summary={'source':str(directory),'status_counts':dict(status),'blocks':len(blocks),
             'complete_face_blocks':sum(r['all_four_probes_optimum'] for r in faces),
             'certified_unavoidable_continuity_losses':sum(r.get('all_weighted_optima_have_worse_continuity',False) for r in faces),
             'certified_unavoidable_overtime_losses':sum(r.get('all_weighted_optima_have_worse_overtime',False) for r in faces),
             'certified_unavoidable_both_losses':sum(r.get('all_weighted_optima_have_worse_both',False) for r in faces),
             'weighted_faces_with_continuity_variation':sum(r.get('weighted_face_has_continuity_variation',False) for r in faces),
             'weighted_faces_with_overtime_variation':sum(r.get('weighted_face_has_overtime_variation',False) for r in faces),
             'verified_budget_steps':len(checks),
             'budget_instances_with_overtime_reduction':len(budget_overtime_instances),
             'total_overtime_reduction_across_adjacent_budgets':sum(r['overtime_reduction'] for r in checks),
             'all_runs_optimum':bool(status) and set(status)=={'OPTIMUM'},
             'interpretation':'CONT and OT intervals are separate extrema; endpoints need not occur in the same schedule. Incomplete or timed-out probes support no optimal-face claim. Budget k limits additional aggregate continuity penalty, not the number of affected patients.'}
    write_json(output/'policy_diagnostics_validation.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir',type=Path);p.add_argument('output_dir',type=Path)
    a=p.parse_args();result=analyze(a.result_dir,a.output_dir);print(result)
    raise SystemExit(0 if result['all_runs_optimum'] and result['complete_face_blocks']==result['blocks'] else 2)
