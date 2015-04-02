#!/usr/bin/env python
#
# Dan Levin <dan@net.t-labs.tu-berlin.de>

import networkx as nx
import os
import sys
import unittest

# set up include path for direct test invocation during development
sys.path.append(os.path.dirname(__file__) + "/..")

from sim.topology import EnterpriseTopology
from sim.enterprisedata import EnterpriseData

class TestEnterpriseTopology(unittest.TestCase):
    """ Tests for the classes of the EnterpriseTopology class"""

    def setUp(self):
        self.topo = EnterpriseTopology.load_from_pickle("2004", "max-50", 1, 2)
        #self.pd = EnterpriseData()

    @unittest.skip("Can't test because of missing enterprise data")
    def test_graph_invariants(self):
        """ EnterpriseTopology graph must maintain the following invariants """

        raw_links = self.pd.links()


        l2adjacencies = sum([len(d.l2adjacencies()) for d in self.pd.devices()])
        self.assertEqual(l2adjacencies,
            4201), "There must be 4201 directed links with capacity >= 0 in the raw enterprisedata dataset"

        self.assertEqual(len(self.pd.links()),
            3912), "There must be 3912 directed links with capacity > 0 in the raw enterprisedata dataset"


    def test_port_api(self):
        """ Test that traffic volume is accounted for consistently through the
        Port and EnterpriseTopology class apis"""
        topo = self.topo

        volume1 = topo.endpoint_volume_from_node('config1515')

        volume2 = sum([x.volume() for x in topo.ports_on_switch('config1515')])

        self.assertEqual(volume1, volume2)

    def test_flow_conservation(self):
        """Test that traffic volume into a switch + from endpoints must =
        traffic out + to endpoionts"""
        topo = self.topo

        for node in topo.graph.nodes():
            volin = sum([x[0] for x in topo.egress_volumes_at_node(node)]) + topo.endpoint_volume_to_node(node)
            volout = sum([x[0] for x in topo.ingress_volumes_at_node(node)]) + topo.endpoint_volume_from_node(node)
            assert abs(volin - volout) < 5

    @unittest.skip("Can't test because of missing enterprise data")
    def test_max_50_base_load_projection(self):
        """Test that the max_50 base load projection does not overload any
        link, as links have different capacities"""

        # iterate over each of the seedmapping values
        for seed in range(1, 11):
            EnterpriseTopology(mappingseed=seed)
            # topo will fail an assertion if any link is overloaded


    @unittest.skip("Can't test because of missing enterprise data")
    def test_1_50_base_load_projection(self):
        """Test that the 1_50 base load projection does not overload any link,
        as links have different capacities"""

        with self.assertRaisesRegexp(AssertionError, 
                                     "No acceptable scaling factor found"):
        # iterate over each of the seedmapping values
            for seed in range(1, 11):
                EnterpriseTopology(tm_factor_name="1-50", mappingseed=seed)
                # topo will fail an assertion if any link is overloaded


    @unittest.skip("Can't test because of missing enterprise data")
    def test_0_5_50_base_load_projection(self):
        """Test that the 0.5-50 base load projection does not overload any
        link, as links have different capacities"""

        with self.assertRaisesRegexp(AssertionError, 
                                     "No acceptable scaling factor found"):
            # iterate over each of the seedmapping values
            for seed in range(1, 11):
                EnterpriseTopology(tm_factor_name="0.5-50", mappingseed=seed)
                # topo will fail an assertion if any link is overloaded

    def test_shortest_path_via(self):
        """Tests if the shortest_path_via function is returning the right paths
        for to simple examples"""

        topo = self.topo
        src = 'config862'

        via = 'config860'
        dst = 'config858'
        path1 = [src, 'config861', via, 'config859', dst]
        path2 = topo.shortest_path_via(src, dst, via)
        self.assertEqual(len(path1), len(path2))
        for x,y in zip(path1,path2):
            self.assertEqual(x, y)

        via = 'config859'
        dst = 'config861'
        path1 = [src, dst, 'config860', via, 'config860', dst]
        path2 = topo.shortest_path_via(src, dst, via)
        self.assertEqual(len(path1), len(path2))
        for x,y in zip(path1,path2):
            self.assertEqual(x, y)

    def test_enterprise_graph_properties(self):
        """ EnterpriseTopology graph must be directed, containt 1577 nodes, be
        strongly connected and the paths need to be symmetric"""

        topo = self.topo

        self.assertTrue(topo.graph.is_directed())
        self.assertEqual(len(topo.graph.nodes()) + len(topo.leaves), 1577)
        self.assertTrue(nx.is_strongly_connected(topo.graph))
        for src in topo.graph.nodes():
            for dst in topo.graph.nodes():
                self.assertEqual(topo.paths[src][dst],
                                 list(reversed(topo.paths[dst][src])))

if __name__ == '__main__':
    unittest.main()
