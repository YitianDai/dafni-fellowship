# Copyright (c) 2026 Yitian Dai <yitian.dai@manchester.ac.uk>
# All rights reserved.

# rundcopf_mp.py
# Drop-in replacement for pypower.rundcopf that uses dcopf_solver_mp
# (translated from MATPOWER 8.0) instead of pypower's dcopf_solver.
#
# Usage:
#   from rundcopf_mp import rundcopf_mp
#   result = rundcopf_mp(ppc, ppopt)
#
# The only difference from rundcopf() is the DC solver:
#   - Correct one-sided bound initialisation (fixes convergence near bounds)
#   - MU_ST from mu_l[Pf]  (MATPOWER convention)

import pypower.opf_execute as _opf_execute_mod
from pypower.ppoption import ppoption
from pypower.opf     import opf

from dcopf_solver_mp import dcopf_solver_mp


def rundcopf_mp(casedata, ppopt=None, fname='', solvedcase=''):
    """Run a DC OPF using the MATPOWER-translated solver (dcopf_solver_mp).

    Parameters
    ----------
    casedata  : pypower case dict
    ppopt     : pypower options dict (ppoption()); PF_DC is forced True
    fname     : (unused, kept for API compatibility)
    solvedcase: (unused, kept for API compatibility)

    Returns
    -------
    result dict  (same structure as pypower rundcopf)
    """
    ppopt = ppoption(ppopt, PF_DC=True)

    # Temporarily swap pypower's dcopf_solver for our translated version.
    # opf_execute imports dcopf_solver at module level, so we patch the
    # reference inside that module for the duration of this call.
    _orig = _opf_execute_mod.dcopf_solver
    _opf_execute_mod.dcopf_solver = dcopf_solver_mp
    try:
        result = opf(casedata, ppopt)
    finally:
        _opf_execute_mod.dcopf_solver = _orig   # always restore

    return result
