#!/usr/bin/env python3
"""
Script to validate dependency compatibility
before building containers.
"""

import os
import sys
from typing import Dict


def parse_requirements_file(file_path: str) -> Dict[str, str]:
    """Extract pinned requirements from a file.

    Returns a mapping package->version (version may be empty string if not pinned).
    Resolves `-r other.txt` includes relative to the current file.
    """
    requirements: Dict[str, str] = {}
    base_dir = os.path.dirname(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue

                # Handle includes (-r other.txt)
                if line.startswith('-r'):
                    included = line[2:].strip()
                    included_path = os.path.join(base_dir, included)
                    # Avoid infinite recursion; only parse if file exists
                    if os.path.exists(included_path):
                        included_reqs = parse_requirements_file(included_path)
                        # preserve existing entries (do not overwrite)
                        for k, v in included_reqs.items():
                            if k not in requirements:
                                requirements[k] = v
                    else:
                        print(f"WARNING: Included requirements file not found: {included_path}")
                    continue

                # Only look for simple pinned 'pkg==version' specs
                if '==' in line:
                    pkg, _, version = line.partition('==')
                    pkg = pkg.strip()
                    version = version.strip()
                    if pkg and version:
                        # preserve first appearance
                        if pkg not in requirements:
                            requirements[pkg] = version
                    else:
                        # malformed line
                        print(f"WARNING: Skipping malformed requirement line in {file_path}: {raw.rstrip()}")
                else:
                    # Non-pinned requirement: record with empty version if first seen
                    pkg = line.split()[0]
                    if pkg not in requirements:
                        requirements[pkg] = ''

    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
    except Exception as exc:
        print(f"ERROR: Error reading {file_path}: {exc}")

    return requirements


def validate_compatibility() -> bool:
    """Validates compatibility between requirements files.

    Success criteria: no package is pinned to different versions across the target files.
    """
    print("VALIDATING DEPENDENCY COMPATIBILITY")

    base_reqs = parse_requirements_file("src/requirements/base.txt")
    api_reqs = parse_requirements_file("src/requirements/api.txt")
    airflow_reqs = parse_requirements_file("docker/airflow/requirements.txt")

    conflicts = []

    # helper to compare two maps and collect conflicts
    def compare_maps(map_a: Dict[str, str], name_a: str, map_b: Dict[str, str], name_b: str):
        for pkg, ver_a in map_a.items():
            if pkg in map_b:
                ver_b = map_b[pkg]
                # only consider a conflict when both are pinned and different
                if ver_a and ver_b and ver_a != ver_b:
                    conflicts.append(f"CONFLICT: {pkg} {name_a}={ver_a} vs {name_b}={ver_b}")

    compare_maps(base_reqs, 'base', api_reqs, 'api')
    compare_maps(base_reqs, 'base', airflow_reqs, 'airflow')
    compare_maps(api_reqs, 'api', airflow_reqs, 'airflow')

    if conflicts:
        print("ERROR: CONFLICTS FOUND:")
        for conflict in conflicts:
            print(f"  {conflict}")
        return False

    print("SUCCESS: ALL DEPENDENCIES ARE SUPPORTED")
    return True


if __name__ == "__main__":
    ok = validate_compatibility()
    if not ok:
        sys.exit(1)