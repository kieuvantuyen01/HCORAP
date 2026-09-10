#!/usr/bin/env python3
"""Materialize separate, reproducible pilot/full campaign configurations (no solver runs)."""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/configs'


def relative(path):return os.path.relpath(path,CONFIG)


def write(name,config):
    config['result_dir']=f'../results/{name}'
    (CONFIG/f'{name}.json').write_text(json.dumps(config,indent=2)+'\n')


def prepare():
    source=json.loads((CONFIG/'gcp_commercial_corrected_primary.json').read_text())
    directory=ROOT/'instances/corrected_v2_reduced_suite/evaluation_critical/evaluation/critical'
    parents=sorted(p for p in directory.glob('*.txt') if any(f'_seed{s}_' in p.name for s in (1001,1002,1003)))
    pilot=sorted(directory/p.name.replace('seed1001','seed1002') if ('u30_a15_v5' in p.name or 'u30_a25_v4' in p.name) else p
                 for p in parents if '_seed1001_' in p.name)
    assert len(parents)==48 and len(pilot)==16
    common={k:v for k,v in source.items() if k not in ('result_dir','instances','instance_filters','expected_instances','expected_runs','runs')}
    common.update(order_seed=20260909,workers=1,threads=1)
    anchors=[{'method':method,'print_assignments':True,'native_log':True} for method in ('weighted','lex-cos','lex-overtime')]
    probes=[{'method':'weighted-face','probe_objective':obj,'probe_sense':sense,'print_assignments':True,'native_log':True} for obj in ('continuity','overtime') for sense in ('min','max')]
    budgets=[{'method':'continuity-budget','continuity_slack':k,'print_assignments':True,'native_log':True} for k in (0,1,2)]
    weights=[{'method':'weighted','wc':wc,'wo':wo,'print_assignments':True,'native_log':True} for wc in (1,4,8) for wo in (1,4,8)]
    for scale,paths in [('pilot',pilot),('full',parents)]:
        for name,runs in [('diagnostics',anchors+probes+budgets),('weights',weights)]:
            write(f'research_depth_{name}_{scale}',{**common,'instances':[relative(p) for p in paths],
                 'expected_instances':len(paths),'expected_runs':len(paths)*len(runs),'runs':runs})
        load_paths=[ROOT/'instances/research_depth_load_sweep'/f'rho{rho:g}_normal{f:g}'/p.name for rho in (.55,.85,.98) for f in (.70,.85,1.) for p in paths]
        if not all(p.is_file() for p in load_paths):raise ValueError('run generate_load_sweep.py first')
        write(f'research_depth_load_{scale}',{**common,'instances':[relative(p) for p in load_paths],
             'expected_instances':len(load_paths),'expected_runs':len(load_paths)*3,'runs':anchors})
    audit=json.loads((CONFIG/'research_depth_diagnostics_pilot.json').read_text())
    audit['commercial_configurations']=[{'backend':'cplex-mip','formulation':'mip-e'}]
    write('research_depth_diagnostics_cplex_audit',audit)
    original=json.loads((CONFIG/'gcp_original_policy_encoding_3600.json').read_text())
    original['instance_filters']={'seeds':[1]}
    original['expected_instances']=16;original['expected_runs']=64
    original['order_seed']=20260909
    original['runs']=[{'variant':f'zero-continuity-{name}','method':'lex-cos','align_evalmaxsat_tct':True,
                       'print_assignments':True,'zero_continuity_local':enabled}
                      for name,enabled in [('global',False),('local',True)]]
    write('research_depth_zero_continuity_pilot',original)
    local={**original,'instances':['../../instances/paperInstances/TXT_10-25_4-5_U30/instance_30_10_4_1.txt'],
           'expected_instances':1,'expected_runs':4,'timeout_seconds':120}
    local.pop('instance_filters')
    write('research_depth_zero_continuity_local_check',local)
    smoke={**common,'instances':['../../tests/instances/tradeoff.txt','../../tests/instances/lex_cos_tie.txt'],
           'commercial_configurations':[{'backend':'reference-enumerator','formulation':'direct-schedule-enumeration'}],
           'timeout_seconds':10,'expected_instances':2,'expected_runs':20,
           'runs':[{k:v for k,v in r.items() if k!='native_log'} for r in anchors+probes+budgets]}
    write('research_depth_smoke',smoke)
    write('research_depth_weights_smoke',{**smoke,'expected_runs':18,'runs':[{k:v for k,v in r.items() if k!='native_log'} for r in weights]})
    # Full evaluation includes the pilot cells. Campaigns keep separate output
    # and identities; use full only after inspecting the pilot, without merging
    # duplicates as independent observations.
    print('Created 11 configurations; pilot covers all 3 known COS/OCS conflicts. No solver launched.')


if __name__=='__main__':prepare()
