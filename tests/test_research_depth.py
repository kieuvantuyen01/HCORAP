from __future__ import annotations
import itertools
import json
import subprocess
from dataclasses import replace
from pathlib import Path
import pytest

from hcorap.io import read_instance, write_instance
from hcorap.model import Assignment
from hcorap.metrics import verify_assignments
from experiments.analyze_policy_structure import describe
from experiments.generate_load_sweep import generate
from experiments.research_depth_common import sha256, resolve_instance
from experiments.run_commercial_campaign import _validate_run

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'bin/release/hcorap_commercial'


def test_gcp_orchestrator_has_valid_shell_and_documents_guarded_phases():
    script = ROOT / 'experiments/run_research_depth_gcp.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    completed = subprocess.run(
        ['bash', str(script), 'help'], check=True, capture_output=True, text=True
    )
    assert '960 driver runs' in completed.stdout
    assert '2,208 Gurobi runs' in completed.stdout
    assert 'CONFIRM_RESEARCH_DEPTH_PILOT=YES' in completed.stdout
    assert 'CONFIRM_RESEARCH_DEPTH_FULL=YES' in completed.stdout


def test_gcp_orchestrator_reported_campaign_sizes_match_frozen_configs():
    config_dir = ROOT / 'experiments/configs'
    pilot = [
        'research_depth_diagnostics_pilot.json',
        'research_depth_diagnostics_cplex_audit.json',
        'research_depth_weights_pilot.json',
        'research_depth_load_pilot.json',
        'research_depth_zero_continuity_pilot.json',
    ]
    full = [
        'research_depth_diagnostics_full.json',
        'research_depth_weights_full.json',
        'research_depth_load_full.json',
    ]
    total = lambda names: sum(
        json.loads((config_dir / name).read_text())['expected_runs']
        for name in names
    )
    assert total(pilot) == 960
    assert total(full) == 2208


def run(path,method,*extra):
    result=subprocess.run([str(BINARY),str(path),'--backend','reference-enumerator','--method',method,
                           '--timeout','10','--print-assignments',*extra],capture_output=True,text=True,check=True,timeout=15)
    return json.loads(result.stdout)


def feasible_metrics(instance):
    candidates=[[Assignment(a,s,t) for a,t in instance.candidate_triplets(s)] for s in range(instance.services)]
    return [v.metrics for schedule in itertools.product(*candidates)
            if (v:=verify_assignments(instance,schedule)).valid]


@pytest.mark.parametrize('wc,wo',[(1,0),(1,1),(20,20),(0,0)])
@pytest.mark.parametrize('objective',['continuity','overtime'])
@pytest.mark.parametrize('sense',['min','max'])
def test_weighted_face_matches_independent_enumeration(wc,wo,objective,sense):
    path=ROOT/'tests/instances/tradeoff.txt';instance=read_instance(path)
    feasible=feasible_metrics(instance)
    score=lambda m:m.similarity-wc*m.continuity_penalty-wo*instance.penalty*m.overtime
    optimum=max(map(score,feasible));face=[m for m in feasible if score(m)==optimum]
    value=lambda m:m.continuity_penalty if objective=='continuity' else m.overtime
    expected=(min if sense=='min' else max)(map(value,face))
    result=run(path,'weighted-face','--wc',str(wc),'--wo',str(wo),'--probe-objective',objective,'--probe-sense',sense)
    assert result['status']=='OPTIMUM'
    assert result['weighted_optimum_reference']==optimum
    assert result['metrics']['weighted_reference_score']==optimum
    assert result['metrics'][objective]==expected
    assert result['solver_calls']==2
    assert verify_assignments(instance,tuple(Assignment(*a) for a in result['assignments'])).valid


@pytest.mark.parametrize('slack',[0,1,3])
@pytest.mark.parametrize('name',['tradeoff.txt','lex_cos_tie.txt'])
def test_continuity_budget_matches_independent_enumeration(slack,name):
    path=ROOT/'tests/instances'/name;instance=read_instance(path)
    feasible=feasible_metrics(instance);anchor=min(m.continuity_penalty for m in feasible)
    allowed=[m for m in feasible if m.continuity_penalty<=anchor+slack]
    expected=min((m.overtime,-m.similarity) for m in allowed)
    result=run(path,'continuity-budget','--continuity-slack',str(slack))
    assert result['status']=='OPTIMUM'
    assert result['continuity_optimum_reference']==anchor
    assert result['metrics']['continuity']<=anchor+slack
    assert (result['metrics']['overtime'],-result['metrics']['similarity'])==expected
    if slack==0:
        reference=run(path,'lex-cos')
        assert all(result['metrics'][k]==reference['metrics'][k] for k in ('continuity','overtime','similarity'))


@pytest.mark.parametrize('spec',[
    {'method':'continuity-budget','continuity_slack':-1},
    {'method':'continuity-budget','continuity_slack':.5},
    {'method':'continuity-budget','continuity_slack':True},
    {'method':'weighted-face','probe_objective':'similarity'},
    {'method':'weighted-face','probe_sense':'up'},
    {'method':'weighted-face','soft_coverage':True},
])
def test_diagnostic_config_rejects_invalid_semantics(spec):
    with pytest.raises(ValueError):_validate_run(spec)


def test_schedule_structure_measures_groups_and_overtime():
    instance=read_instance(ROOT/'tests/instances/tradeoff.txt')
    result=run(ROOT/'tests/instances/tradeoff.txt','lex-cos')
    desc,*_=describe(instance,tuple(Assignment(*a) for a in result['assignments']))
    assert desc['split_groups']==0
    assert desc['agents_with_overtime']==1


def test_archive_resolution_checks_content_hash():
    path=ROOT/'instances/corrected_v2_reduced_suite/evaluation_critical/evaluation/critical/instance_u30_a10_v4_seed1001_critical.txt'
    foreign='/remote/HCORAP/'+str(path.relative_to(ROOT))
    assert resolve_instance(foreign,sha256(path))==path
    with pytest.raises(ValueError,match='hash mismatch'):resolve_instance(foreign,'wrong')


def test_load_sweep_changes_only_capacity_and_reproduces_anchor(tmp_path):
    parent=ROOT/'instances/corrected_v2_reduced_suite/evaluation_critical/evaluation/critical/instance_u30_a10_v4_seed1001_critical.txt'
    result=generate([(parent,sha256(parent))],tmp_path,rhos=(.85,.98),normal_fractions=(.70,.85))
    assert result['variants']==4
    assert sum(x['anchor'] for x in result['instances'])==1
    original=read_instance(parent)
    for row in result['instances']:
        new=read_instance(Path(row['instance']))
        assert replace(new,normal_hours=original.normal_hours,extra_hours=original.extra_hours,source=original.source)==original
        assert row['witness_verified']
    with pytest.raises(ValueError,match='already contains'):generate([(parent,sha256(parent))],tmp_path)


def test_backend_comparison_checks_probe_objective_not_arbitrary_ties():
    from experiments.collect_commercial_campaign import backend_agreement
    base={'instance_sha256':'same','instance':'tiny','method':'weighted-face','delta':'-',
          'wc':1,'wo':0,'soft_coverage':False,'status':'OPTIMUM','probe_objective':'continuity',
          'probe_sense':'min','weighted_optimum_reference':8,'weighted_reference_score':8,
          'coverage':2,'similarity':8,'continuity':0,'overtime':0,'backend':'gurobi-mip','formulation':'mip-e'}
    other={**base,'backend':'cplex-mip','overtime':1}
    rows=backend_agreement([base,other])
    assert rows[0]['certified_objective_agreement'] is True
    assert rows[0]['objective_vector_agreement'] is False


def test_diagnostic_campaign_keeps_probe_and_budget_identities_distinct(tmp_path):
    from experiments.run_commercial_campaign import _build_tasks
    config_path=ROOT/'experiments/configs/research_depth_smoke.json'
    config=json.loads(config_path.read_text())
    configs=[dict(c,resolved_parameter_file=None) for c in config['commercial_configurations']]
    tasks=_build_tasks(config,base=config_path.parent,binary_hash='test',commercial_configs=configs)
    assert len(tasks)==len({t['run_id'] for t in tasks})==20
    cells={(t['specification']['method'],t['specification'].get('probe_objective'),t['specification'].get('probe_sense'),t['specification'].get('continuity_slack')) for t in tasks}
    assert len(cells)==10


def test_diagnostic_analysis_rejects_tampered_anchor(tmp_path):
    from experiments.run_commercial_campaign import run_campaign
    from experiments.analyze_policy_diagnostics import analyze
    config=json.loads((ROOT/'experiments/configs/research_depth_smoke.json').read_text())
    config.update(binary=str(BINARY),instances=[str(ROOT/'tests/instances/tradeoff.txt')],expected_instances=1,expected_runs=10,result_dir=str(tmp_path/'result'))
    config_path=tmp_path/'campaign.json';config_path.write_text(json.dumps(config))
    assert run_campaign(config_path)['complete']
    result=analyze(tmp_path/'result',tmp_path/'analysis')
    assert result['complete_face_blocks']==1
    assert result['verified_budget_steps']==2
    for path in (tmp_path/'result/raw').glob('*.json'):
        payload=json.loads(path.read_text())
        if payload['method']=='weighted-face':
            payload['weighted_optimum_reference']+=1;path.write_text(json.dumps(payload));break
    with pytest.raises(ValueError,match='score mismatch'):analyze(tmp_path/'result',tmp_path/'bad-analysis')


def test_cross_backend_budget_signature_ignores_unoptimized_continuity():
    from experiments.compare_policy_diagnostics import signature
    a={'method':'continuity-budget','continuity_optimum_reference':0,
       'metrics':{'continuity':0,'overtime':0,'similarity':8,'coverage':2}}
    b={**a,'metrics':{**a['metrics'],'continuity':1}}
    assert signature(a)==signature(b)


def test_zero_continuity_collector_preserves_failure_without_speedup(tmp_path):
    from experiments.analyze_zero_continuity import analyze
    raw=tmp_path/'raw';raw.mkdir()
    records=[]
    for local in (False,True):
        name=str(local)
        payload={'method':'lex-cos','timeout_seconds':120,'status':'ERROR','error':'signal 11',
                 'stages':[{'objective':'continuity','status':'OPTIMUM','optimum':0,'hard_clauses':100},
                           {'objective':'overtime','status':'OPTIMUM','optimum':0,'hard_clauses':20 if local else 100},
                           {'objective':'similarity','status':'ERROR','hard_clauses':20 if local else 100}]}
        (raw/f'{name}.json').write_text(json.dumps(payload))
        records.append({'run_id':name,'instance':'tiny','instance_sha256':'hash','result_status':'ERROR',
                        'specification':{'method':'lex-cos','cardinality':'totalizer','zero_continuity_local':local}})
    (tmp_path/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (tmp_path/'validation.json').write_text(json.dumps({'complete':True,'expected_runs':2}))
    result=analyze(tmp_path,tmp_path/'analysis')
    assert result['both_optimum_pairs']==0
    assert result['status_counts']=={'ERROR':2}
    assert 'global_over_local_elapsed' not in (tmp_path/'analysis/zero_continuity_pairs.csv').read_text()
