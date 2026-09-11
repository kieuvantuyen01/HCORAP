#!/usr/bin/env python3
"""Generate paired capacity variants from the archived 48-instance HCORAP-LC suite."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_depth_common import ROOT, campaign_records, resolve_instance, sha256, write_json
sys.path.insert(0, str(ROOT / 'src/proposed'))
from hcorap.io import read_instance
from hcorap.generator import calibrate_capacity, generation_witness, write_generated_instance
from hcorap.metrics import verify_assignments


PUBLIC_POLICY_TABLE = ROOT / 'artifact/results/policy/corrected_pairwise_pairs.csv'


def public_parents(table: Path = PUBLIC_POLICY_TABLE) -> list[tuple[Path, str]]:
    """Load the 48 parent instances from the versioned artifact table."""
    parents: dict[Path, str] = {}
    with table.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row['instance']
            digest = row['instance_sha256']
            previous = parents.setdefault(path, digest)
            if previous != digest:
                raise ValueError(f'inconsistent hashes for {path}')
    if len(parents) != 48:
        raise ValueError(f'expected 48 public parents, found {len(parents)}')
    return sorted(parents.items())


def generate(parents: list[tuple[Path,str]], output: Path, rhos=(.55,.85,.98), normal_fractions=(.70,.85,1.0)):
    if (output/'load_sweep_manifest.json').exists() or any(output.glob('*/*.txt')):
        raise ValueError(f'output already contains a sweep: {output}')
    rows=[]
    for path,parent_hash in sorted(parents):
        if sha256(path)!=parent_hash: raise ValueError(f'parent hash mismatch: {path}')
        sidecar=path.with_suffix(path.suffix+'.json')
        metadata=json.loads(sidecar.read_text())['metadata']
        parent=replace(read_instance(path),metadata=metadata)
        witness=generation_witness(parent)
        if not witness or not verify_assignments(parent,witness).valid:
            raise ValueError(f'missing/invalid full coverage witness: {path}')
        for rho in rhos:
            for fraction in normal_fractions:
                child=calibrate_capacity(parent,target_rho=rho,normal_fraction=fraction)
                if not verify_assignments(child,witness).valid:
                    raise ValueError(f'capacity variant loses witness feasibility: {path}')
                # Everything except normal/extra capacity and provenance must match.
                if replace(child,normal_hours=parent.normal_hours,extra_hours=parent.extra_hours,metadata=parent.metadata)!=parent:
                    raise AssertionError('non-capacity instance field changed')
                tag=f'rho{rho:g}_normal{fraction:g}'
                target=output/tag/path.name
                provenance={'parent_instance':str(path.relative_to(ROOT)), 'parent_sha256':parent_hash,
                            'parent_sidecar_sha256':sha256(sidecar),'rho':rho,'normal_fraction':fraction}
                child=replace(child,metadata={**child.metadata,'load_sweep':provenance})
                write_generated_instance(child,target)
                anchor=rho==.85 and fraction==.85
                if anchor and sha256(target)!=parent_hash:
                    raise ValueError(f'anchor does not reproduce parent bytes: {path}')
                rows.append({**provenance,'instance':str(target.resolve()),'instance_sha256':sha256(target),
                             'anchor':anchor,'realized_rho':child.metadata['capacity_calibration']['realized_rho'],
                             'witness_verified':True})
    result={'schema_version':1,'parents':len(parents),'variants':len(rows),
            'rhos':list(rhos),'normal_fractions':list(normal_fractions),
            'scope':'Feasibility-preserving paired capacity sweep. Actual rho may differ from requested rho. Parent instances share nested families; do not treat all cells as independent replicates.',
            'instances':rows}
    write_json(output/'load_sweep_manifest.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,help='optional archived corrected campaign')
    p.add_argument('--output',type=Path,default=ROOT/'instances/research_depth_load_sweep')
    a=p.parse_args()
    if a.source is None:
        parent_rows = public_parents()
    else:
        parents = {}
        for record, payload in campaign_records(a.source):
            path = resolve_instance(record['instance'], record['instance_sha256'])
            parents[path] = record['instance_sha256']
        parent_rows = list(parents.items())
    result=generate(parent_rows,a.output)
    print(f"Verified {result['variants']} variants from {result['parents']} parents; manifest: {a.output/'load_sweep_manifest.json'}")
