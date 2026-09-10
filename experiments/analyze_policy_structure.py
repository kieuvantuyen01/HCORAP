#!/usr/bin/env python3
"""Reverify archived schedules and compare their patient/group/workload structure."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_depth_common import ROOT, campaign_records, resolve_instance, write_csv, write_json
sys.path.insert(0, str(ROOT / 'src/proposed'))
from hcorap.io import read_instance
from hcorap.metrics import verify_assignments
from hcorap.model import Assignment


def describe(instance, assignments):
    checked = verify_assignments(instance, assignments)
    if not checked.valid:
        raise ValueError(checked.violations)
    agent = {a.service: a.agent for a in assignments}
    slot = {a.service: a.time_slot for a in assignments}
    groups = [set(agent[s] for s in sequence) for sequence in instance.sequences]
    loads = [sum(a.agent == i for a in assignments) for i in range(instance.agents)]
    metrics = checked.metrics
    return {
        'continuity': metrics.continuity_penalty, 'overtime': metrics.overtime,
        'similarity': metrics.similarity, 'coverage': metrics.coverage,
        'groups': len(groups), 'split_groups': sum(len(g) > 1 for g in groups),
        'max_caregivers_per_group': max(map(len, groups), default=0),
        'agents_with_overtime': metrics.agents_with_overtime,
        'max_agent_overtime': metrics.max_agent_overtime,
        'max_agent_workload': max(loads, default=0),
    }, agent, slot, groups, loads


def analyze(directory: Path, output: Path):
    rows, group_rows, workload_rows = [], [], []
    paired = defaultdict(dict)
    skipped = defaultdict(int)
    for record, payload in campaign_records(directory):
        if payload['status'] != 'OPTIMUM':
            skipped[payload['status']] += 1
            continue
        if 'assignments' not in payload:
            raise ValueError(f'missing assignments: {record["run_id"]}')
        path = resolve_instance(record['instance'], record['instance_sha256'])
        instance = read_instance(path)
        desc, agents, slots, groups, loads = describe(instance, tuple(Assignment(*a) for a in payload['assignments']))
        for key in ('coverage', 'similarity', 'continuity', 'overtime'):
            if desc[key] != payload['metrics'][key]:
                raise ValueError(f'metric mismatch: {record["run_id"]}:{key}')
        spec = record['specification']
        base = {'run_id': record['run_id'], 'instance': str(path.relative_to(ROOT)),
                'instance_sha256': record['instance_sha256'], 'method': payload['method'],
                'backend': payload.get('backend', 'maxsat'), 'wc': spec.get('wc', 1), 'wo': spec.get('wo', 1)}
        rows.append({**base, **desc})
        for q, (sequence, caregivers) in enumerate(zip(instance.sequences, groups)):
            group_rows.append({**base, 'group': q, 'services': len(sequence),
                               'caregivers': len(caregivers), 'split': len(caregivers) > 1,
                               'caregiver_ids': '/'.join(map(str, sorted(caregivers)))})
        for a, load in enumerate(loads):
            workload_rows.append({**base, 'agent': a, 'workload': load,
                                  'normal_capacity': instance.normal_hours[a],
                                  'extra_capacity': instance.extra_hours[a],
                                  'overtime': max(0, load-instance.normal_hours[a])})
        # Pair only equal solver/config/weights, avoid mixing independent experiments.
        key = (base['instance_sha256'], base['backend'], base['wc'], base['wo'],
               payload.get('cardinality_encoding'), payload.get('implied_constraints'), payload.get('symmetry_breaking'))
        if payload['method'] in paired[key]:
            raise ValueError(f'duplicate policy for pairing: {key}')
        paired[key][payload['method']] = (base, desc, agents, slots, groups, loads)
    pairs = []
    for policies in paired.values():
        for left, right in combinations(sorted(policies), 2):
            base, a, aa, at, ag, aw = policies[left]
            _, b, ba, bt, bg, bw = policies[right]
            pairs.append({**{k:v for k,v in base.items() if k not in ('method','run_id')},
                          'left': left, 'right': right,
                          **{f'{key}_{side}': item[key] for side,item in [('left',a),('right',b)] for key in ('continuity','overtime','similarity','split_groups')},
                          'different_caregiver_services': sum(aa[s] != ba[s] for s in aa),
                          'different_slot_services': sum(at[s] != bt[s] for s in at),
                          'different_caregiver_groups': sum(x != y for x,y in zip(ag,bg)),
                          'workload_l1': sum(abs(x-y) for x,y in zip(aw,bw))})
    conflicts = [r for r in pairs if (r['left'],r['right']) == ('lex-cos','lex-overtime') and
                 any(r[f'{k}_left'] != r[f'{k}_right'] for k in ('continuity','overtime','similarity'))]
    for name, data in [('policy_structure',rows),('group_structure',group_rows),('agent_workload',workload_rows),('policy_structure_pairs',pairs),('cos_ocs_conflicts',conflicts)]:
        write_csv(output / f'{name}.csv',data)
    summary = {'source': str(directory), 'independently_verified_optima': len(rows),
               'skipped_statuses': dict(skipped), 'policy_pairs': len(pairs),
               'cos_ocs_conflicts': len(conflicts),
               'interpretation': 'Assignment differences describe returned schedules; they do not prove unavoidable differences between optimal solution sets.'}
    write_json(output/'policy_structure_summary.json',summary)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir',type=Path); parser.add_argument('output_dir',type=Path)
    args=parser.parse_args(); print(analyze(args.result_dir,args.output_dir))
