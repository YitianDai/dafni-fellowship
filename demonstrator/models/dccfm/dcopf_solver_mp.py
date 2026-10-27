# Copyright (c) 2026 Yitian Dai <yitian.dai@manchester.ac.uk>
# All rights reserved.

# Translated from MATPOWER 8.0 dcopf_solver.m to Python.
# Key improvements over pypower's dcopf_solver:
#   1. Uses scipy.optimize.linprog (HiGHS backend) instead of PIPS.
#      HiGHS is a proper simplex/IPM LP solver that handles degenerate vertices
#      correctly — PIPS fails when the optimum has many active bounds simultaneously
#      (e.g. all DL generators fully served, typical with low LOAD_SHED_COST).
#   2. Correct one-sided bound initialisation matching MATPOWER (unused by HiGHS
#      but kept for future solver compatibility).
#   3. MU_ST is taken from mu_l[Pf] (MATPOWER convention) not mu_u[Pt].

from copy import deepcopy
from math import inf

import numpy as np
from numpy import array, zeros, ones, diag, r_, pi, isnan, arange, dot
from numpy import c_ as np_c_
from numpy import flatnonzero as find
from scipy.sparse import vstack, hstack, csr_matrix as sparse
from scipy.optimize import linprog, LinearConstraint, Bounds

from pypower.idx_bus  import BUS_TYPE, REF, VA, LAM_P, LAM_Q, MU_VMAX, MU_VMIN
from pypower.idx_gen  import PG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pypower.idx_brch import PF, PT, QF, QT, RATE_A, MU_SF, MU_ST
from pypower.idx_cost import MODEL, POLYNOMIAL, PW_LINEAR, NCOST, COST

from pypower.util import sub2ind


def dcopf_solver_mp(om, ppopt, out_opt=None):
    """DC OPF solver using scipy HiGHS LP backend.

    Drop-in replacement for pypower.dcopf_solver. Uses scipy.optimize.linprog
    with the HiGHS solver (available in scipy >= 1.7) which reliably handles
    degenerate LP optima that cause PIPS to fail.

    Parameters
    ----------
    om     : pypower opf_model object
    ppopt  : pypower options dict (from ppoption())
    out_opt: ignored, kept for signature compatibility

    Returns
    -------
    results, success, raw  (same structure as pypower dcopf_solver)
    """
    # ---------- unpack data ----------
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = (
        ppc['baseMVA'], ppc['bus'], ppc['gen'], ppc['branch'], ppc['gencost']
    )
    cp = om.get_cost_params()
    N, H, Cw = cp['N'], cp['H'], cp['Cw']
    fparm = array(np_c_[cp['dd'], cp['rh'], cp['kk'], cp['mm']])
    Bf    = om.userdata('Bf')
    Pfinj = om.userdata('Pfinj')
    vv, ll, _, _ = om.get_idx()

    # ---------- problem dimensions ----------
    ipol  = find(gencost[:, MODEL] == POLYNOMIAL)
    nb    = bus.shape[0]
    nl    = branch.shape[0]
    nw    = N.shape[0]
    ny    = om.getN('var', 'y')
    nxyz  = om.getN('var')

    # ---------- linear constraints & variable bounds ----------
    A, l_con, u_con = om.linear_constraints()
    _, xmin, xmax   = om.getv()

    # ---------- build objective CC vector (linear costs only for DC OPF) ----------
    any_pwl = int(ny > 0)
    if any_pwl:
        Npwl      = sparse((ones(ny), (zeros(ny), arange(vv['i1']['y'], vv['iN']['y']))), (1, nxyz))
        Hpwl      = sparse((1, 1))
        Cpwl      = array([1.0])
        fparm_pwl = array([[1, 0, 0, 1]])
    else:
        Npwl      = None
        Hpwl      = None
        Cpwl      = array([])
        fparm_pwl = zeros((0, 4))

    npol = len(ipol)
    iqdr   = find(gencost[ipol, NCOST] == 3)
    ilin   = find(gencost[ipol, NCOST] == 2)
    polycf = zeros((npol, 3))
    if len(iqdr) > 0:
        polycf[iqdr, :]   = gencost[ipol[iqdr], COST:COST + 3]
    if npol:
        polycf[ilin, 1:3] = gencost[ipol[ilin], COST:COST + 2]
    polycf = dot(polycf, diag([baseMVA**2, baseMVA, 1]))

    if npol:
        Npol = sparse((ones(npol), (arange(npol), vv['i1']['Pg'] + ipol)), (npol, nxyz))
        Hpol = sparse((2 * polycf[:, 0], (arange(npol), arange(npol))), (npol, npol))
    else:
        Npol = None
        Hpol = None
    Cpol      = polycf[:, 1]
    fparm_pol = ones((npol, 1)) * array([[1, 0, 0, 1]])

    NN    = vstack([n for n in [Npwl, Npol, N] if n is not None and n.shape[0] > 0], 'csr')
    if (Hpwl is not None) and any_pwl and (npol + nw):
        Hpwl = hstack([Hpwl, sparse((any_pwl, npol + nw))])
    if Hpol is not None:
        if any_pwl and npol:
            Hpol = hstack([sparse((npol, any_pwl)), Hpol])
        if npol and nw:
            Hpol = hstack([Hpol, sparse((npol, nw))])
    if (H is not None) and nw and (any_pwl + npol):
        H = hstack([sparse((nw, any_pwl + npol)), H])
    HHw    = vstack([h for h in [Hpwl, Hpol, H] if h is not None and h.shape[0] > 0], 'csr')
    CCw    = r_[Cpwl, Cpol, Cw]
    ffparm = r_[fparm_pwl, fparm_pol, fparm]

    nnw = any_pwl + npol + nw
    M   = sparse((ffparm[:, 3], (range(nnw), range(nnw))))
    MR  = M * ffparm[:, 1]
    HMR = HHw * MR
    MN  = M * NN
    HH  = MN.T * HHw * MN
    CC  = MN.T * (CCw - HMR)
    C0  = 0.5 * dot(MR, HMR) + sum(polycf[:, 2])

    # DC OPF with polynomial costs is a pure LP (HH is zero or diagonal quadratic).
    # Extract the linear cost vector; handle any small quadratic diagonal (generators
    # with c2 > 0) by noting HH is diagonal — we still pass it to linprog as a QP
    # via the identity: minimize c'x + 0.5 x'Hx. scipy.linprog only does LP so we
    # keep the quadratic term only if it's truly zero, otherwise fall back gracefully.

    CC_dense = np.asarray(CC).flatten()

    # Check if problem is pure LP (HH == 0) or has quadratic terms
    HH_dense = np.asarray(HH.todense()) if hasattr(HH, 'todense') else np.asarray(HH)
    has_quadratic = np.any(np.abs(HH_dense) > 1e-10)

    if has_quadratic:
        # Fall back to PIPS for quadratic (AC-like) costs
        from pypower.qps_pypower import qps_pypower
        from pypower.util import have_fcn
        alg = ppopt['OPF_ALG_DC']
        if alg == 0:
            alg = 200
        opt_pips = {'alg': alg, 'verbose': ppopt['VERBOSE']}
        # MATPOWER initial point
        Varefs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180.0)
        lb = xmin.copy(); ub = xmax.copy()
        lb[xmin == -inf] = -1e10;  ub[xmax == inf] = 1e10
        x0 = (lb + ub) / 2.0
        k = find((xmin == -inf) & (xmax < inf));  x0[k] = xmax[k] - 1.0
        k = find((xmin > -inf) & (xmax == inf));  x0[k] = xmin[k] + 1.0
        x0[vv['i1']['Va']:vv['iN']['Va']] = Varefs[0]
        feastol = ppopt['PDIPM_FEASTOL'] or ppopt['OPF_VIOLATION']
        opt_pips['pips_opt'] = {
            'feastol': feastol, 'gradtol': ppopt['PDIPM_GRADTOL'],
            'comptol': ppopt['PDIPM_COMPTOL'], 'costtol': ppopt['PDIPM_COSTTOL'],
            'max_it': ppopt['PDIPM_MAX_IT'], 'max_red': ppopt['SCPDIPM_RED_IT'],
            'cost_mult': 1,
        }
        x, f_val, info, output, lmbda = qps_pypower(HH, CC, A, l_con, u_con, xmin, xmax, x0, opt_pips)
        success = (info == 1)
        f_val = f_val + C0
    else:
        # Pure LP — use scipy HiGHS
        # Convert equality/inequality constraints:
        # pypower gives l_con <= A*x <= u_con
        # linprog wants: A_ub*x <= b_ub  and  A_eq*x == b_eq
        l_finite = np.isfinite(l_con)
        u_finite = np.isfinite(u_con)

        # equality: l == u (both finite and equal)
        eq_mask = l_finite & u_finite & (np.abs(u_con - l_con) < 1e-10)
        # inequality rows: split into upper and lower bounds
        ub_mask  = u_finite & ~eq_mask
        lb_mask  = l_finite & ~eq_mask

        A_csr = A.tocsr()

        rows_eq  = A_csr[eq_mask,  :]
        rows_ub  = A_csr[ub_mask,  :]
        rows_lb  = A_csr[lb_mask,  :]   # l <= Ax  →  -Ax <= -l

        from scipy.sparse import vstack as sp_vstack
        A_ub = sp_vstack([rows_ub, -rows_lb], format='csr') if (rows_ub.shape[0] + rows_lb.shape[0]) > 0 else None
        b_ub = np.concatenate([u_con[ub_mask], -l_con[lb_mask]])
        A_eq = rows_eq if rows_eq.shape[0] > 0 else None
        b_eq = u_con[eq_mask] if rows_eq.shape[0] > 0 else None

        # variable bounds as list of (lb, ub) tuples — linprog convention
        lb_vars = np.where(np.isfinite(xmin), xmin, -np.inf)
        ub_vars = np.where(np.isfinite(xmax), xmax,  np.inf)
        bounds  = list(zip(lb_vars.tolist(), ub_vars.tolist()))

        # HiGHS dual simplex first; on a non-optimal termination (status 15 "Unknown" occurs on large,
        # badly scaled cases such as the 2000-bus GB network) retry the same LP with HiGHS interior point,
        # then with presolve off.  The LP is unchanged; only the algorithm differs.
        for _method, _opts in (('highs-ds', {}), ('highs-ipm', {}), ('highs-ds', {'presolve': False})):
            res = linprog(
                c       = CC_dense,
                A_ub    = A_ub,
                b_ub    = b_ub,
                A_eq    = A_eq,
                b_eq    = b_eq,
                bounds  = bounds,
                method  = _method,
                options = {'disp': False, **_opts},
            )
            if res.status == 0 or res.status == 2:   # optimal, or genuinely infeasible: stop
                break

        x       = res.x if res.x is not None else zeros(nxyz)
        f_val   = res.fun + C0 if res.fun is not None else 0.0
        success = res.status == 0   # 0 = optimal

        # Reconstruct dual variables from HiGHS output
        # res.ineqlin.marginals: duals for A_ub rows (sign: shadow price of <= constraint)
        # res.eqlin.marginals:   duals for A_eq rows
        n_con = A_csr.shape[0]
        mu_u_full = zeros(n_con)
        mu_l_full = zeros(n_con)

        if success:
            # equality duals
            if A_eq is not None and hasattr(res, 'eqlin') and res.eqlin.marginals is not None:
                eq_idx  = np.where(eq_mask)[0]
                eq_dual = -res.eqlin.marginals   # linprog convention: negate for min
                # positive dual → upper active, negative → lower active
                mu_u_full[eq_idx] = np.maximum( eq_dual, 0)
                mu_l_full[eq_idx] = np.maximum(-eq_dual, 0)

            # inequality duals
            if A_ub is not None and hasattr(res, 'ineqlin') and res.ineqlin.marginals is not None:
                ineq_duals = -res.ineqlin.marginals
                n_ub = rows_ub.shape[0]
                n_lb = rows_lb.shape[0]
                ub_idx = np.where(ub_mask)[0]
                lb_idx = np.where(lb_mask)[0]
                mu_u_full[ub_idx] = np.maximum( ineq_duals[:n_ub], 0)
                mu_l_full[lb_idx] = np.maximum(-ineq_duals[n_ub:], 0)

            # variable bound duals
            if hasattr(res, 'lower') and res.lower.marginals is not None:
                muLB = np.maximum(-res.lower.marginals, 0)
            else:
                muLB = zeros(nxyz)
            if hasattr(res, 'upper') and res.upper.marginals is not None:
                muUB = np.maximum(-res.upper.marginals, 0)
            else:
                muUB = zeros(nxyz)
        else:
            muLB = zeros(nxyz)
            muUB = zeros(nxyz)

        lmbda = {
            'mu_l':  mu_l_full,
            'mu_u':  mu_u_full,
            'lower': muLB,
            'upper': muUB,
        }
        info   = 1 if success else 0
        output = {'alg': 'highs', 'status': res.status, 'message': res.message}

    # ---------- extract primal results ----------
    if success and not any(isnan(x)):
        Va = x[vv['i1']['Va']:vv['iN']['Va']]
        Pg = x[vv['i1']['Pg']:vv['iN']['Pg']]

        bus[:, VA]  = Va * 180.0 / pi
        gen[:, PG]  = Pg * baseMVA

        branch[:, [QF, QT]] = zeros((nl, 2))
        branch[:, PF] = (Bf * Va + Pfinj) * baseMVA
        branch[:, PT] = -branch[:, PF]

    mu_l_v = lmbda['mu_l']
    mu_u_v = lmbda['mu_u']
    muLB   = lmbda['lower']
    muUB   = lmbda['upper']

    il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
    bus[:,    [LAM_P, LAM_Q, MU_VMIN, MU_VMAX]]     = zeros((nb, 4))
    gen[:,    [MU_PMIN, MU_PMAX, MU_QMIN, MU_QMAX]] = zeros((gen.shape[0], 4))
    branch[:, [MU_SF, MU_ST]]                        = zeros((nl, 2))

    bus[:, LAM_P]     = (mu_u_v[ll['i1']['Pmis']:ll['iN']['Pmis']] -
                         mu_l_v[ll['i1']['Pmis']:ll['iN']['Pmis']]) / baseMVA
    branch[il, MU_SF] = mu_u_v[ll['i1']['Pf']:ll['iN']['Pf']] / baseMVA
    branch[il, MU_ST] = mu_l_v[ll['i1']['Pf']:ll['iN']['Pf']] / baseMVA
    gen[:, MU_PMIN]   = muLB[vv['i1']['Pg']:vv['iN']['Pg']] / baseMVA
    gen[:, MU_PMAX]   = muUB[vv['i1']['Pg']:vv['iN']['Pg']] / baseMVA

    pimul = r_[
        mu_l_v - mu_u_v,
        -ones(int(ny > 0)),
        muLB - muUB,
    ]

    mu = {'var': {'l': muLB, 'u': muUB},
          'lin': {'l': mu_l_v, 'u': mu_u_v}}

    results = deepcopy(ppc)
    results['bus'], results['branch'], results['gen'], \
        results['om'], results['x'], results['mu'], results['f'] = \
        bus, branch, gen, om, x, mu, f_val

    raw = {'xr': x, 'pimul': pimul, 'info': info, 'output': output}

    return results, success, raw
