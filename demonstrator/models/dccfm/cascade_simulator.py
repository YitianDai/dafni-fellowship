# Copyright (c) 2026 Yitian Dai <yitian.dai@manchester.ac.uk>
# All rights reserved.

"""
DC cascade failure simulator

Dependencies
------------
    numpy, networkx, pypower

"""

import copy
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from pypower.api import rundcopf as _rundcopf_default, ppoption
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD
from pypower.idx_brch import (F_BUS, T_BUS, RATE_A, BR_STATUS, PF, QF, PT, QT)
from pypower.idx_gen import (GEN_BUS, PG, QG, QMIN, QMAX,
                              RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q,
                              VG, MBASE, GEN_STATUS, PMAX, PMIN)
from pypower.idx_cost import NCOST
from pypower.isload import isload


# ---------------------------------------------------------------------------
# load2disp: convert bus loads to dispatchable negative-Pg generators
# ---------------------------------------------------------------------------
def load2disp(ppc: dict, load_shed_cost: float = 5000.0) -> dict:
    """
    Convert fixed bus loads into dispatchable load generators (MATPOWER convention).

    Each load bus gets one negative-Pg generator appended to ppc['gen']:
        PMIN = -PD  (full load served)
        PMAX =  0   (full load shed)
        cost = load_shed_cost + bus_id * 0.01  (small perturbation breaks degeneracy)

    The original bus PD/QD is zeroed to avoid double-counting.
    Appends 'DL' / 'dl' to gentype / genfuel when those fields are present.
    """
    ppc = copy.deepcopy(ppc)
    bus       = ppc['bus']
    gen_cols  = ppc['gen'].shape[1]
    cost_cols = ppc['gencost'].shape[1]
    new_gens, new_costs = [], []

    for i in np.where(bus[:, PD] != 0)[0]:
        bus_id = int(bus[i, BUS_I])
        pd_val = float(bus[i, PD])
        qd_val = float(bus[i, QD])

        g = np.zeros(gen_cols)
        g[GEN_BUS]    = bus_id
        g[PG]         = -pd_val
        g[QG]         = -qd_val
        g[QMIN]       = -qd_val
        g[QMAX]       = 0.0
        g[VG]         = bus[i, 7]
        g[MBASE]      = ppc['baseMVA']
        g[GEN_STATUS] = 1
        g[PMAX]       = 0
        g[PMIN]       = -pd_val
        g[[RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q]] = np.inf
        new_gens.append(g)

        n_coeffs = cost_cols - 4
        if n_coeffs not in (2, 3):
            raise ValueError("Unsupported gencost polynomial width.")
        coeffs = [0.0, load_shed_cost + bus_id * 0.01, 0.0][-n_coeffs:]
        new_costs.append([2, 0, 0, n_coeffs] + coeffs)
        bus[i, [PD, QD]] = 0.0

    if new_gens:
        n = len(new_gens)
        ppc['gen']     = np.vstack([ppc['gen'],     np.asarray(new_gens)])
        ppc['gencost'] = np.vstack([ppc['gencost'], np.asarray(new_costs)])
        if 'gentype' in ppc and ppc['gentype'] is not None:
            ppc['gentype'] = list(ppc['gentype']) + ['DL'] * n
        if 'genfuel' in ppc and ppc['genfuel'] is not None:
            ppc['genfuel'] = list(ppc['genfuel']) + ['dl'] * n
    return ppc


# ---------------------------------------------------------------------------
# load2disp_dummy_gen: add a dummy positive-Pg generator beside each load
# ---------------------------------------------------------------------------
def load2disp_dummy_gen(ppc: dict, load_shed_cost: float = 5000.0) -> dict:
    """
    Model load shedding by adding a dummy generator at each load bus instead
    of converting the load to a negative-Pg generator.

    Each load bus gets one additional generator appended to ppc['gen']:
        PMIN = 0        (no injection by default)
        PMAX = PD       (can inject up to the full load, i.e. fully shed it)
        PG   = 0
        cost = load_shed_cost + bus_id * 0.01  (perturbation breaks degeneracy, matching load2disp)

    The bus PD is left unchanged — the OPF sees the original load and the
    dummy generator offsets it when shedding.  This maps directly to the
    pandapower pattern:

        pp.create_sgen(net, bus, p_mw=0, min_p_mw=0, max_p_mw=load_p_mw,
                       controllable=True)
        pp.create_poly_cost(net, idx, et='sgen',
                            cp1_eur_per_mw=load_shed_cost,
                            cp2_eur_per_mw2=1)
    """
    ppc = copy.deepcopy(ppc)
    bus       = ppc['bus']
    gen_cols  = ppc['gen'].shape[1]
    cost_cols = ppc['gencost'].shape[1]
    new_gens, new_costs = [], []

    for i in np.where(bus[:, PD] != 0)[0]:
        bus_id = int(bus[i, BUS_I])
        pd_val = float(bus[i, PD])

        g = np.zeros(gen_cols)
        g[GEN_BUS]    = bus_id
        g[PG]         = 0.0
        g[QG]         = 0.0
        g[VG]         = bus[i, 7]
        g[MBASE]      = ppc['baseMVA']
        g[GEN_STATUS] = 1
        g[PMAX]       = pd_val   # max injection = full load (fully shed)
        g[PMIN]       = 0.0      # min injection = 0 (no shedding)
        g[[RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q]] = np.inf
        new_gens.append(g)

        # cost form matches the existing gencost polynomial width
        n_coeffs = cost_cols - 4
        if n_coeffs not in (2, 3):
            raise ValueError("Unsupported gencost polynomial width.")
        # coeffs order (MATPOWER): [c2, c1, c0] trimmed to n_coeffs
        # bus_id*0.01 perturbation breaks degeneracy, matching load2disp
        coeffs = [0.0, load_shed_cost + bus_id * 0.01, 0.0][-n_coeffs:]
        new_costs.append([2, 0, 0, n_coeffs] + coeffs)

    if new_gens:
        n = len(new_gens)
        ppc['gen']     = np.vstack([ppc['gen'],     np.asarray(new_gens)])
        ppc['gencost'] = np.vstack([ppc['gencost'], np.asarray(new_costs)])
        if 'gentype' in ppc and ppc['gentype'] is not None:
            ppc['gentype'] = list(ppc['gentype']) + ['DL'] * n
        if 'genfuel' in ppc and ppc['genfuel'] is not None:
            ppc['genfuel'] = list(ppc['genfuel']) + ['dl'] * n
    return ppc


# ---------------------------------------------------------------------------
# disp2load: reverse of load2disp — restore bus loads from dispatchable gens
# ---------------------------------------------------------------------------
def disp2load(ppc: dict) -> dict:
    """
    Reverse load2disp: convert dispatchable negative-Pg generators back to
    fixed bus loads (MATPOWER AC-CFM convention).

    Uses pypower.isload (Pmin < 0 and Pmax == 0) to identify load rows.
    Adds -Pg back to bus PD and -Qg back to bus QD (accumulates rather than
    overwrites, matching MATPOWER's disp2load behaviour), then removes those
    rows from gen, gencost, gentype and genfuel (when present).
    """
    ppc = copy.deepcopy(ppc)
    bus = ppc['bus']
    gen = ppc['gen']

    load_mask     = isload(gen)
    load_rows     = gen[load_mask]
    bus_id_to_row = {int(bus[r, BUS_I]): r for r in range(len(bus))}

    for row in load_rows:
        br = bus_id_to_row[int(row[GEN_BUS])]
        bus[br, PD] -= float(row[PG])   # PG < 0, so PD increases
        bus[br, QD] -= float(row[QG])

    keep = ~load_mask
    ppc['gen']     = gen[keep]
    ppc['gencost'] = ppc['gencost'][keep]
    if 'gentype' in ppc and ppc['gentype'] is not None:
        ppc['gentype'] = [v for v, k in zip(ppc['gentype'], keep) if k]
    if 'genfuel' in ppc and ppc['genfuel'] is not None:
        ppc['genfuel'] = [v for v, k in zip(ppc['genfuel'], keep) if k]
    return ppc


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Output of CascadeSimulator.run()."""
    network_after:        dict
    ovld_lines_log: list            # list of arrays, one per cascade iteration
    total_load_shed:      float           # total demand loss (MW)
    n_active_islands:     list            # active island count per iteration
    n_all_islands:        list            # total island count per iteration


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------
class CascadeSimulator:
    """
    DC cascade failure simulator.

    Each iteration: per-island redispatch OPF (with ramp headroom and load shedding),
    OFGT to trip the smallest generator(s) on excess generation, then overloaded-line
    tripping to advance the cascade.

    Parameters
    ----------
    ppc : dict
        PyPower case (MATPOWER-compatible).  Loads must be encoded as
        negative-Pg generators with PMIN < 0 and PMAX = 0.
    gen_redispatch_headroom : ndarray of shape (ng,), optional
        Per-generator fraction by which Pg can be adjusted up or down in the
        redispatch OPF.  Default 0.99 for all generators.
    load_shed_headroom : ndarray of shape (ng,), optional
        Per-load fraction of current Pg that may be shed in the redispatch
        OPF.  Default 0.99 for all loads.
    verbose : bool
        Print iteration messages (default True).
    solver : callable, optional
        DC OPF solver function with signature ``solver(ppc, ppopt) -> result``.
        Defaults to ``rundcopf_mp`` (scipy HiGHS backend).
        Pass ``pypower.api.rundcopf`` to use the original PIPS solver instead.
    """

    def __init__(
        self,
        ppc: dict,
        gen_redispatch_headroom: Optional[np.ndarray] = None,
        load_shed_headroom: Optional[np.ndarray] = None,
        verbose: bool = True,
        solver=None,
    ):
        self.ppc = ppc
        ng = ppc['gen'].shape[0]

        self.gen_redispatch_headroom = (
            gen_redispatch_headroom if gen_redispatch_headroom is not None
            else np.full(ng, 0.99)
        )
        self.load_shed_headroom = (
            load_shed_headroom if load_shed_headroom is not None
            else np.full(ng, 0.99)
        )
        self.verbose = verbose
        self._ppopt  = ppoption(VERBOSE=0, OUT_ALL=0, PF_DC=1)

        if solver is None:
            from rundcopf_mp import rundcopf_mp
            self._solver = rundcopf_mp
        else:
            self._solver = solver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        faulty_lines: np.ndarray,
        faulty_nodes: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Run the cascade simulation.

        Parameters
        ----------
        faulty_lines : array-like of int
            0-based row indices of branches to disconnect.
        faulty_nodes : array-like of int, optional
            Bus numbers (BUS_I values) to trip.

        Returns
        -------
        SimulationResult
        """
        faulty_lines = np.asarray(faulty_lines, dtype=int)
        faulty_nodes = (np.asarray(faulty_nodes, dtype=int)
                        if faulty_nodes is not None else np.array([], dtype=int))

        network = copy.deepcopy(self.ppc)

        # --- Initialise state ---
        ovld_lines_log  = []
        ovld_lines_prev = np.array([], dtype=int)   # overloaded lines from previous iteration

        n_active_islands    = []
        n_all_islands       = []
        n_blackout_islands = 0

        # Apply initial faults
        network['branch'][faulty_lines, BR_STATUS] = 0
        network, _ = self._trip_nodes(network, faulty_nodes)

        for i in range(int(1e9)):

            if self.verbose:
                fl_pairs = [
                    f"{int(network['branch'][k, F_BUS])}-{int(network['branch'][k, T_BUS])}"
                    for k in faulty_lines
                ]
                print(f'\n{"=" * 44}')
                print(f'  Cascade iteration {i + 1}')
                print(f'{"=" * 44}')
                print(f'  Initial faults : {" ".join(fl_pairs)}')
                if len(ovld_lines_prev) > 0:
                    tp_pairs = [
                        f"{int(network['branch'][k, F_BUS])}-{int(network['branch'][k, T_BUS])}"
                        for k in ovld_lines_prev
                    ]
                    print(f'  Prev overloaded: {" ".join(tp_pairs)}')

            # Per-iteration accumulators
            tripped_nodes_iter   = np.array([], dtype=int)
            tripped_lines_iter   = np.array([], dtype=int)
            ofgt_tripped_gens    = np.array([], dtype=int)   # bus numbers (for messages)
            ofgt_tripped_rows    = np.array([], dtype=int)   # row indices into network['gen'] (for state update)
            ovld_lines_iter = np.array([], dtype=int)

            # Ensure every island has a reference bus
            network = self._add_reference_bus(network)

            # net_fixed_dispatch: generators pinned to current dispatch, line limits removed
            net_fixed_dispatch = copy.deepcopy(network)
            net_fixed_dispatch['branch'][:, RATE_A] = 0

            gen_mask  = network['gen'][:, PMAX] > 0
            load_mask = network['gen'][:, PMIN] < 0
            net_fixed_dispatch['gen'][gen_mask,  PMIN] = network['gen'][gen_mask,  PG]
            net_fixed_dispatch['gen'][gen_mask,  PMAX] = network['gen'][gen_mask,  PG]
            net_fixed_dispatch['gen'][load_mask, PMIN] = network['gen'][load_mask, PG]
            net_fixed_dispatch['gen'][load_mask, PMAX] = network['gen'][load_mask, PG]

            # net_active_buses: inactive buses/branches removed, used for island detection
            net_active_buses = copy.deepcopy(net_fixed_dispatch)
            active_mask = net_active_buses['bus'][:, BUS_TYPE] != 4
            net_active_buses['bus'] = net_active_buses['bus'][active_mask]
            active_bus_ids = set(net_active_buses['bus'][:, BUS_I].astype(int))
            br = net_active_buses['branch']
            net_active_buses['branch'] = br[
                np.isin(br[:, F_BUS].astype(int), list(active_bus_ids)) &
                np.isin(br[:, T_BUS].astype(int), list(active_bus_ids))
            ]

            islands = self._find_islands(net_active_buses)

            if self.verbose:
                print(f'  Islands found  : {len(islands)}')
                print(f'  {"-" * 42}')
            n_all_islands.append(len(islands) + n_blackout_islands)

            # ---- process each island ----------------------------------------
            for isl_j, isl_row_idx in enumerate(islands):
                isl_bus_ids = net_active_buses['bus'][isl_row_idx, BUS_I].astype(int)

                if self.verbose:
                    print(f'\n  Island {isl_j + 1}: buses [ {" ".join(str(n) for n in isl_bus_ids)} ]')

                isl_ovld_lines = np.array([], dtype=int)

                # Build island sub-network from net_fixed_dispatch
                island_net = copy.deepcopy(net_fixed_dispatch)
                island_net['bus'][:, BUS_TYPE] = 4
                in_island_mask = np.isin(island_net['bus'][:, BUS_I].astype(int), isl_bus_ids)
                island_net['bus'][in_island_mask, BUS_TYPE] = net_fixed_dispatch['bus'][in_island_mask, BUS_TYPE]

                # Zero generators outside island (before removing rows)
                gen_in_island_mask = np.isin(island_net['gen'][:, GEN_BUS].astype(int), isl_bus_ids)
                island_net['gen'][~gen_in_island_mask, :] = 0

                # Remove off-island rows
                island_net['gencost'] = island_net['gencost'][gen_in_island_mask]
                if 'gentype' in island_net and island_net['gentype'] is not None:
                    island_net['gentype'] = [v for v, k in zip(island_net['gentype'], gen_in_island_mask) if k]
                if 'genfuel' in island_net and island_net['genfuel'] is not None:
                    island_net['genfuel'] = [v for v, k in zip(island_net['genfuel'], gen_in_island_mask) if k]
                island_net['gen']     = island_net['gen'][gen_in_island_mask]
                isl_rows = np.where(gen_in_island_mask)[0]       # network['gen'] row of each island_net['gen'] row

                # gen_snapshot: [bus, Pg, Pmax, Pmin, redispatch_headroom, shed_headroom]
                _gm = np.isin(network['gen'][:, GEN_BUS].astype(int), isl_bus_ids)
                gen_snapshot = np.column_stack([
                    network['gen'][_gm, GEN_BUS],
                    network['gen'][_gm, PG],
                    network['gen'][_gm, PMAX],
                    network['gen'][_gm, PMIN],
                    self.gen_redispatch_headroom[_gm],
                    self.load_shed_headroom[_gm],
                ])

                island_load_ini = float(
                    -np.sum(island_net['gen'][island_net['gen'][:, PG] < 0, PG])
                )

                # --- First OPF (generators locked to current dispatch) ------
                opf_result = self._run_opf(island_net)

                if opf_result['success'] == 1:
                    if self.verbose:
                        print('    OPF status   : success (no redispatch)')
                else:
                    # Relax generator and load bounds for redispatch
                    is_gen  = gen_snapshot[:, 2] > 0   # Pmax > 0 = real generators
                    is_load = gen_snapshot[:, 3] < 0   # Pmin < 0 = loads

                    # Generators: allow ramp down to (1-headroom)*Pg and up to (1+headroom)*Pg,
                    # but clamp within original [Pmin, Pmax]
                    island_net['gen'][is_gen, PMIN] = np.maximum(
                        gen_snapshot[is_gen, 3],
                        (1 - gen_snapshot[is_gen, 4]) * gen_snapshot[is_gen, 1],
                    )
                    island_net['gen'][is_gen, PMAX] = np.minimum(
                        gen_snapshot[is_gen, 2],
                        (1 + gen_snapshot[is_gen, 4]) * gen_snapshot[is_gen, 1],
                    )
                    # Loads: allow shed up to shed_headroom fraction of current Pg;
                    # Pmax is relaxed upward (less negative), Pmin fixed at current Pg
                    island_net['gen'][is_load, PMAX] = (
                        (1 - gen_snapshot[is_load, 5]) * gen_snapshot[is_load, 1]
                    )
                    island_net['gen'][is_load, PMIN] = gen_snapshot[is_load, 1]

                    # imbalance > 0: excess generation; imbalance < 0: deficit
                    imbalance     = float(np.sum(island_net['gen'][:, PG]))
                    gen_rows      = island_net['gen'][:, PMAX] > 0
                    load_rows     = island_net['gen'][:, PMIN] < 0

                    # sum(Pmax_gen) - sum(Pg_gen) > 0: headroom to ramp up
                    cap_up_gen    = (np.sum(island_net['gen'][gen_rows,  PMAX])
                                     - np.sum(island_net['gen'][gen_rows,  PG]))
                    # sum(Pmin_gen) - sum(Pg_gen) < 0: headroom to ramp down
                    cap_down_gen  = (np.sum(island_net['gen'][gen_rows,  PMIN])
                                     - np.sum(island_net['gen'][gen_rows,  PG]))
                    # -sum(Pmin_load) + sum(Pg_load) ~0: load increase headroom (Pmin pinned to Pg)
                    cap_up_load   = (-np.sum(island_net['gen'][load_rows, PMIN])
                                     + np.sum(island_net['gen'][load_rows, PG]))
                    # -sum(Pmax_load) + sum(Pg_load) < 0: load shed headroom (Pmax less negative than Pg)
                    cap_down_load = (-np.sum(island_net['gen'][load_rows, PMAX])
                                     + np.sum(island_net['gen'][load_rows, PG]))
                    # net upward flexibility: gen ramp-up + load shedding
                    cap_up        = cap_up_gen   - cap_down_load
                    # net downward flexibility: gen ramp-down - load increase
                    cap_down      = cap_down_gen - cap_up_load

                    has_load = np.sum(island_net['gen'][load_rows, PG]) < 0

                    if imbalance > 0 and not has_load:
                        # excess generation but no load — blackout immediately
                        if self.verbose:
                            print('    OPF status   : failed — no load in island — island blacked out')
                        opf_result = {'success': 0}

                    elif imbalance > 0 and (imbalance + cap_down) > 0:
                        # OFGT: excess generation with load present, ramp-down alone insufficient
                        if self.verbose:
                            print('    OFGT         : excess generation detected')
                        while (imbalance > 0 and (imbalance + cap_down) > 0):
                            pos_pg_rows = np.where(island_net['gen'][:, PMAX] > 0)[0]
                            # Positive PG generators only
                            pg_rounded = np.round(island_net['gen'][pos_pg_rows, PG], 1)

                            # Minimum PG after rounding to 1 decimal place
                            min_pg_val = np.min(pg_rounded)

                            # Choose the first generator in pos_pg_rows with this rounded PG value
                            min_pos = np.where(pg_rounded == min_pg_val)[0][0]
                            min_pg_row = pos_pg_rows[min_pos]

                            # Corresponding bus number
                            bus_num = int(island_net['gen'][min_pg_row, GEN_BUS])
                            if self.verbose:
                                print(f'    OFGT         : gen @ bus {bus_num} tripped')
                            ofgt_tripped_gens = np.append(ofgt_tripped_gens, bus_num)
                            ofgt_tripped_rows = np.append(ofgt_tripped_rows, isl_rows[min_pg_row])
                            isl_rows          = np.delete(isl_rows, min_pg_row)
                            # remove tripped generator from all parallel arrays to keep them in sync
                            island_net['gen']     = np.delete(island_net['gen'],     min_pg_row, axis=0)
                            island_net['gencost'] = np.delete(island_net['gencost'], min_pg_row, axis=0)
                            if 'gentype' in island_net and island_net['gentype'] is not None:
                                island_net['gentype'] = [v for j, v in enumerate(island_net['gentype']) if j != min_pg_row]
                            if 'genfuel' in island_net and island_net['genfuel'] is not None:
                                island_net['genfuel'] = [v for j, v in enumerate(island_net['genfuel']) if j != min_pg_row]
                            gen_snapshot          = np.delete(gen_snapshot,           min_pg_row, axis=0)
                            imbalance     = float(np.sum(island_net['gen'][:, PG]))
                            gen_rows      = island_net['gen'][:, PMAX] > 0
                            load_rows     = island_net['gen'][:, PMIN] < 0
                            cap_up_gen    = (np.sum(island_net['gen'][gen_rows,  PMAX])
                                             - np.sum(island_net['gen'][gen_rows,  PG]))
                            cap_down_gen  = (np.sum(island_net['gen'][gen_rows,  PMIN])
                                             - np.sum(island_net['gen'][gen_rows,  PG]))
                            cap_up_load   = (-np.sum(island_net['gen'][load_rows, PMIN])
                                             + np.sum(island_net['gen'][load_rows, PG]))
                            cap_down_load = (-np.sum(island_net['gen'][load_rows, PMAX])
                                             + np.sum(island_net['gen'][load_rows, PG]))
                            cap_up        = cap_up_gen   - cap_down_load
                            cap_down      = cap_down_gen - cap_up_load

                        # after OFGT: check if a generation deficit was created
                        if imbalance < 0 and (imbalance + cap_up) < 0:
                            if self.verbose:
                                print('    OPF status   : failed — OFGT caused generation deficit — island blacked out')
                            opf_result = {'success': 0}
                        else:
                            opf_result = self._run_opf(island_net)

                    elif imbalance < 0 and (imbalance + cap_up) < 0:
                        # generation deficit, not resolvable by ramp-up + load shedding
                        if self.verbose:
                            print('    OPF status   : failed — unresolvable generation deficit — island blacked out')
                        opf_result = {'success': 0}

                    else:
                        # imbalance within resolvable range: cap_down <= -imbalance <= cap_up
                        opf_result = self._run_opf(island_net)

                    if opf_result['success'] == 1:
                        if self.verbose:
                            print('    OPF status   : success (with redispatch)')

                        # Zero out OFGT-tripped generators in main network (by row, so that
                        # other units on the same bus are untouched)
                        network['gen'][ofgt_tripped_rows, PG:QG + 1] = 0
                        network['gen'][ofgt_tripped_rows, GEN_STATUS] = 0

                        # Persist redispatch result: write OPF Pg back to main network
                        # so load shedding and gen re-dispatch carry over to next iteration.
                        # opf_result['gen'] has the same rows, in the same order, as island_net['gen']
                        # (rundcopf_mp restores the external order), and isl_rows maps those rows to
                        # network['gen'].  Matching by row (not by bus) is required when a bus
                        # carries several generators: bus matching gave every unit on a bus the
                        # same Pg, which corrupted the pinned dispatch and bounds in the next iteration.
                        network['gen'][isl_rows, PG] = opf_result['gen'][:, PG]

                        if self.verbose:
                            island_shed = island_load_ini + float(np.sum(
                                opf_result['gen'][opf_result['gen'][:, PG] < 0, PG]
                            ))
                            print(f'    load shed    : {island_shed:.4f} MW')

                    # Blackout island if OPF still fails
                    if opf_result['success'] == 0:
                        n_blackout_islands += 1
                        network, blackout_br_trips = self._trip_nodes(network, isl_bus_ids)
                        tripped_lines_iter = np.unique(
                            np.concatenate([tripped_lines_iter, blackout_br_trips])
                        )
                        tripped_nodes_iter = np.unique(
                            np.concatenate([tripped_nodes_iter, isl_bus_ids])
                        )
                        if self.verbose:
                            print(f'    load shed    : {island_load_ini:.4f} MW')

                # Check line overloading (if OPF converged)
                if opf_result['success'] == 1 and 'branch' in opf_result:
                    br_res  = opf_result['branch']
                    rate_a  = network['branch'][:, RATE_A]
                    flow    = np.maximum(np.abs(br_res[:, PF]), np.abs(br_res[:, PT]))
                    limit   = rate_a * 1.01
                    rated   = rate_a > 0
                    over    = rated & (np.round(flow, 5) > np.round(limit, 5))
                    isl_ovld_lines = np.where(over)[0]

                    if self.verbose and len(isl_ovld_lines) > 0:
                        pairs = [
                            f"{int(network['branch'][k, F_BUS])}-"
                            f"{int(network['branch'][k, T_BUS])}"
                            for k in isl_ovld_lines
                        ]
                        print(f'    Overloaded   : {" ".join(pairs)}')

                ovld_lines_iter = np.unique(
                    np.concatenate([ovld_lines_iter, isl_ovld_lines])
                )
            # ---- end island loop -------------------------------------------

            # Count active islands after this iteration
            active_bus_mask = network['bus'][:, BUS_TYPE] != 4
            if np.any(active_bus_mask):
                net_island_count = copy.deepcopy(network)
                net_island_count['bus'] = net_island_count['bus'][active_bus_mask]
                surviving_bus_ids = set(net_island_count['bus'][:, BUS_I].astype(int))
                br_surviving = net_island_count['branch']
                net_island_count['branch'] = br_surviving[
                    np.isin(br_surviving[:, F_BUS].astype(int), list(surviving_bus_ids)) &
                    np.isin(br_surviving[:, T_BUS].astype(int), list(surviving_bus_ids))
                ]
                n_active_islands.append(len(self._find_islands(net_island_count)))
            else:
                n_active_islands.append(0)

            # Trip overloaded lines or end cascade
            if len(ovld_lines_iter) > 0:
                network['branch'][ovld_lines_iter, BR_STATUS] = 0
                ovld_lines_prev = ovld_lines_iter.copy()
                ovld_lines_log.append(ovld_lines_iter.copy())
                if self.verbose:
                    tripped_pairs = [
                        f"{int(network['branch'][k, F_BUS])}-{int(network['branch'][k, T_BUS])}"
                        for k in ovld_lines_iter
                    ]
                    print(f'\n  {"-" * 42}')
                    print(f'  Active islands : {n_active_islands[-1]}')
                    print(f'  Lines tripped  : {" ".join(tripped_pairs)}')
            else:
                if self.verbose:
                    print(f'\n  {"-" * 42}')
                    print(f'  Active islands : {n_active_islands[-1]}')
                    print('  Cascade stops — no overloads.')
                break  

        # --- Final network state ---
        network_after = copy.deepcopy(network)

        # Recompute total load shed as difference between initial and final load served.
        # This is more reliable than accumulating per-island deltas across iterations.
        load_ini  = -np.sum(self.ppc['gen'][self.ppc['gen'][:, PMIN] < 0, PG])
        load_fin  = -np.sum(network_after['gen'][network_after['gen'][:, PG] < 0, PG])
        total_load_shed = load_ini - load_fin

        if self.verbose:
            print(f'\n  Total load shed: {total_load_shed:.4f} MW  '
                  f'(initial {load_ini:.4f} MW  →  served {load_fin:.4f} MW)')

        return SimulationResult(
            network_after=network_after,
            ovld_lines_log=ovld_lines_log,
            total_load_shed=total_load_shed,
            n_active_islands=n_active_islands,
            n_all_islands=n_all_islands,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _run_opf(self, ppc: dict) -> dict:
        """Run DC OPF; return result dict (always has 'success' key)."""
        try:
            result = self._solver(copy.deepcopy(ppc), self._ppopt)
            return result
        except Exception:
            return {'success': 0}

    def _find_islands(self, ppc: dict):
        """
        Find connected components in the network.

        Returns
        -------
        list of list of int
            Each sub-list is a sorted list of row indices into ppc['bus']
            for one island (single-bus or multi-bus).
        """
        bus       = ppc['bus']
        branch    = ppc['branch']
        n_bus     = len(bus)
        id_to_row = {int(bus[r, BUS_I]): r for r in range(n_bus)}

        G = nx.Graph()
        G.add_nodes_from(range(n_bus))
        for br in branch:
            if int(br[BR_STATUS]) == 1:
                f = int(br[F_BUS])
                t = int(br[T_BUS])
                if f in id_to_row and t in id_to_row:
                    G.add_edge(id_to_row[f], id_to_row[t])

        return [sorted(comp) for comp in nx.connected_components(G)]

    def _add_reference_bus(self, ppc: dict) -> dict:
        """Ensure every island has exactly one reference (slack) bus."""
        ppc      = copy.deepcopy(ppc)
        islands  = self._find_islands(ppc)
        inactive = ppc['bus'][:, BUS_TYPE] == 4

        for isl in islands:
            bus_types = ppc['bus'][isl, BUS_TYPE]
            if not np.any(bus_types == 3):
                ppc['bus'][isl[0], BUS_TYPE] = 3   # first bus becomes slack

        ppc['bus'][inactive, BUS_TYPE] = 4          # restore isolated flag
        return ppc

    def _trip_nodes(self, ppc: dict, nodes: np.ndarray):
        """
        Deactivate buses, zero their loads and generators, trip connected branches.

        Returns
        -------
        ppc : updated network dict
        tripped : array of 0-based branch row indices that were tripped
        """
        ppc = copy.deepcopy(ppc)
        if len(nodes) == 0:
            return ppc, np.array([], dtype=int)

        bus_mask = np.isin(ppc['bus'][:, BUS_I].astype(int), nodes)
        ppc['bus'][bus_mask, BUS_TYPE] = 4
        ppc['bus'][bus_mask, PD]       = 0
        ppc['bus'][bus_mask, QD]       = 0

        gen_mask = np.isin(ppc['gen'][:, GEN_BUS].astype(int), nodes)
        ppc['gen'][gen_mask, :] = 0

        from_mask = np.isin(ppc['branch'][:, F_BUS].astype(int), nodes)
        to_mask   = np.isin(ppc['branch'][:, T_BUS].astype(int), nodes)
        ppc['branch'][from_mask | to_mask, BR_STATUS] = 0

        tripped = np.where(from_mask | to_mask)[0]
        return ppc, tripped
