#!/usr/bin/env python
"""This module contains unit tests for the Simulator and the IterationState
classes."""

import os
import sys
import copy
import unittest

# set up include path for direct test invocation during development
sys.path.append(os.path.dirname(__file__) + "/..")

from sim.topology import EnterpriseTopology
from sim.simulator import Simulator
from panoptisim import get_parser


class TestSimulator(unittest.TestCase):
    """Tests for the Simulator class"""

    def setUp(self):
        self.seed = 1234
        parser = get_parser()
        arg_list = ['new', '--seedmapping', '1', '--seednextswitch', '2',
                    '--tmsf', 'max-50', '--epsf', '2', '--epp', '2',
                    '--tm', '2004', '--switchstrategy', 'VOL',
                    '--portstrategy', 'sdn-switches-first',
                    '--maxvlans', '2048', '--maxft', '100000',
                    '--activefrontiersize', '1000', '--toupgrade', '100']
        arg_list_2 = ['new', '--seedmapping', '1', '--seednextswitch', '2',
                      '--tmsf', 'max-50', '--epsf', '2', '--epp', '2',
                      '--tm', '2004', '--switchstrategy', 'VOL',
                      '--portstrategy', 'sdn-switches-first',
                      '--maxvlans', '2048', '--maxft', '1',
                      '--activefrontiersize', '1000', '--toupgrade', '100']

        self.args = parser.parse_args(arg_list)
        self.args_2 = parser.parse_args(arg_list_2)

    def test_allocate_basic_forwarding_at_sw(self):
        """allocate_basic_forwarding_at_sw needs to be idempotent. This test
        verifies that."""

        topo = EnterpriseTopology.load_from_pickle("2004", "max-50", 1, 2)
        sim = Simulator(topo, self.args)
        istate = sim.istate

        switch = topo.graph.nodes()[0]
        istate.sdn_switches[switch] = topo.graph.node[switch]
        istate.reinitialize()
        port = topo.graph.node[switch]['ports'][0]
        self.assertEqual(istate.allocate_basic_forwarding_at_sw(switch, port),
                         1)
        self.assertEqual(istate.allocate_basic_forwarding_at_sw(switch, port),
                         0)
        self.assertEqual(len(istate.basic_forwarding_rules_at_sw[switch]), 1)

    def test_rollback(self):
        """In case a port could not be made SDN, the current state of the
        simulation needs to be rolled back. This method tests if this is done
        correctly."""

        topo = EnterpriseTopology.load_from_pickle("2004", "max-50", 1, 2)
        sim = Simulator(topo, self.args_2)
        istate = sim.istate

        strategy = "VOL"
        self.assertNotEqual(sim.switch_selection(istate, strategy),
                            0)
        istate.reinitialize()
        istate.frontiers = sim.find_frontiers(istate)
        self.assertEqual(len(istate.remaining_ft_per_switch), 1)

        # Save ramaining ft state
        remaining_ft = {}
        for key in istate.remaining_ft_per_switch.iterkeys():
            remaining_ft[key] = istate.remaining_ft_per_switch[key]

        # Save forwarding rules state
        basic_forwarding_rules_at_sw = {}
        for key in istate.basic_forwarding_rules_at_sw.iterkeys():
            basic_forwarding_rules_at_sw[key] = set()
            for port in istate.basic_forwarding_rules_at_sw[key]:
                basic_forwarding_rules_at_sw[key].add(str(port))

        # Save graph state
        rollback_graph = {}
        for src, dst, edgeattrs in topo.graph.edges(data=True):
            if src not in rollback_graph:
                rollback_graph[src] = {}
            rollback_graph[src][dst] = copy.deepcopy(edgeattrs)

        port = topo.ports()[0]
        sim.make_port_sdn(istate, port, False)

        # Check remaining ft
        for key in istate.remaining_ft_per_switch.iterkeys():
            self.assertEqual(istate.remaining_ft_per_switch[key],
                             remaining_ft[key])

        # Check graph
        for src, dst, edgeattrs in topo.graph.edges(data=True):
            for key in rollback_graph[src][dst]:
                self.assertEqual(edgeattrs[key], rollback_graph[src][dst][key])

        # Checks if reinitialize correctly resets the forwarding rules
        istate.reinitialize()
        for key in istate.basic_forwarding_rules_at_sw.iterkeys():
            for port in istate.basic_forwarding_rules_at_sw[key]:
                self.assertEqual(len(basic_forwarding_rules_at_sw),
                                 len(istate.basic_forwarding_rules_at_sw))
                self.assertIn(str(port), basic_forwarding_rules_at_sw[key])

    def check_istate_reinitialization(self, istate):
        """Checks reinitialization of istate object."""
        self.assertEqual(len(istate.failure_reason), 0)
        for switch in istate.sdn_switches:
            self.assertEqual(istate.remaining_ft_per_switch[switch],
                             istate.args.maxft)
            self.assertEqual(len(istate.basic_forwarding_rules_at_sw[switch]),
                             0)
        for node in istate.graph.nodes():
            self.assertEqual(len(istate.used_vlans_per_switch[node]), 0)
        self.assertEqual(len(istate.sdn_ports), 0)
        for port in istate.topo.ports():
            self.assertEqual(len(port.designated_switch), 0)
            self.assertEqual(len(port.path_to_port), 0)
            self.assertEqual(len(port.path_to_frontiersw), 0)

    def test_reinitialize(self):
        """Checks if IterationState.reinitialize correctly reinitializes the
        iteration state."""

        topo = EnterpriseTopology.load_from_pickle("2004", "max-50", 1, 2)
        sim = Simulator(topo, self.args)
        istate = sim.istate

        istate.reinitialize()
        self.check_istate_reinitialization(istate)

        istate.frontiers = sim.find_frontiers(istate)
        sim.allocate_ft_for_ports_on_sdn_switches(istate)
        sim.make_ports_sdn(istate, cand_ports=istate.ports_on_sdn_switches())

        istate.reinitialize()
        self.check_istate_reinitialization(istate)

if __name__ == '__main__':
    unittest.main()
