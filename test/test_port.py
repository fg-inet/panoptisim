#!/usr/bin/env python
#
# Dan Levin <dan@net.t-labs.tu-berlin.de>

import os
import unittest
import sys

# set up include path for direct test invocation during development
sys.path.append(os.path.dirname(__file__) + "/..")

from sim.topology import EnterpriseTopology

def list_is_true(li):
    """Returns True only if the entire list is true"""
    for i in li:
        if i is not True:
            return i
    return True

@unittest.skip("deprecated due to deprecated 'tm' attribute")
class TestPort(unittest.TestCase):

    """ Tests for the classes of the EnterpriseTopology class"""
    def setUp(self):
        self.topo = EnterpriseTopology()

    def test_graph_invariants(self):
        """ test that the traffic matrix is identical """
        is_true = list_is_true([list_is_true(
            [v == self.topo.graph.node[x.switch]['tm'][int(x.name)][int(k.name)] for k,v
            in x._volume_to.iteritems()]) for x in self.topo.ports()])

        assert is_true is True

if __name__ == '__main__':
    unittest.main()
