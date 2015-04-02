#!/usr/bin/env python
#
# Dan Levin <dan@net.t-labs.tu-berlin.de>

#from topology import EnterpriseTopology
import unittest

class TestHeuristic(unittest.TestCase):
    """ Tests for the classes of the heuristic module """

    def setUp(self):
        pass
#		self.enterprise = EnterpriseTopology("uniform", pruneleaves=True)


	def test_stretch_definition(self):
		""" A Network in which every switch is upgraded should give stretch of 1 """

        pass

if __name__ == '__main__':
	unittest.main()
