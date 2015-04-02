# Dan Levin <dan@net.t-labs.tu-berlin.de>
"""This module contains the Simulator and the IterationState class that the
Simulator uses to track the simulation state."""

from collections import OrderedDict
import gzip
import json
import copy
import logging
import numpy as np
from time import time
import random

import networkx as nx

FORMAT = "%(asctime)s %(process)d %(name)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)


class IterationState(object):
    """Tracks the simulation state that persists over each switch selection
    iteration.

    All of the "SDN state" is kept here:
        self.sdn_ports: the ports that have have their traffic accommodated by
        SDN switches and are subject to "waypoint enforcement"

        self.sdn_switches: the switches in the graph that support an SDN
        interface

    The exception is that Port objects keep a dictionary of their
    designated_switch to a destination port.
    """

    def __init__(self, sim):
        self.sim = sim
        self.args = sim.args
        self.topo = sim.topo
        self.graph = sim.topo.graph

        # Keep the results of every iteration
        self.results = []
        # keep track of why a port failed to be made sdn
        self.failure_reason = {}

        self.previous_sdn_ports = None
        self.previous_sdn_switches = None
        self.previous_remaining_ft_per_switch = None
        self.previous_used_vlans_per_switch = None

        self.sdn_ports = set()
        self.sdn_switches = OrderedDict()
        self.remaining_ft_per_switch = OrderedDict()
        self.basic_forwarding_rules_at_sw = OrderedDict()
        self.used_vlans_per_switch = OrderedDict()

        # all frontiers are indexed by port
        self.frontier_by_component = {}
        self.frontiers = {}
        self.active_frontiers = {}

        self.starttime = None

        # Note that edge traffic-projection state is kept in the graph edges
        # themselves

    def __repr__(self):
        return "Iteration with {} sdn switches".format(len(self.sdn_switches))

    def reinitialize(self):
        """Reinitializes the iteration state after upgrade_switch has been
        called by the Simulator"""

        logger.debug("Reinitializing IterationState")
        self.starttime = time()
        self.failure_reason = {}

        self.previous_sdn_ports = self.sdn_ports
        #self.previous_sdn_switches is reinitialized in upgrade_switch()
        self.previous_remaining_ft_per_switch = self.remaining_ft_per_switch
        self.previous_used_vlans_per_switch = self.used_vlans_per_switch

        # Initialize the remaining flow table state for every switch
        for i in self.sdn_switches:
            self.remaining_ft_per_switch[i] = self.args.maxft
            self.basic_forwarding_rules_at_sw[i] = set()

        # Initialize the vlan state for every switch
        for i in self.graph.nodes():
            self.used_vlans_per_switch[i] = set()

        # Reset port state
        self.sdn_ports = set()
        for port in self.topo.ports():
            port.designated_switch = OrderedDict()
            port.path_to_port = OrderedDict()
            port.path_to_frontiersw = OrderedDict()

        # Reset the topo graph edge state
        self.topo.reinitialize()

    def allocate_basic_forwarding_at_sw(self, switch, port):
        """Idempotently record a basic forwarding entry at switch for port.
        Return the number of ft entries allocated."""
        allocated_entries_per_port = 1
        if port not in self.basic_forwarding_rules_at_sw[switch]:
            self.remaining_ft_per_switch[switch] -= allocated_entries_per_port
            self.basic_forwarding_rules_at_sw[switch].add(port)
            return allocated_entries_per_port
        else:
            return 0

    def upgrade_switch(self, switch):
        """Set the state of a switch and its ports to sdn"""

        self.previous_sdn_switches = self.sdn_switches.copy()
        self.sdn_switches[switch] = True

        if len(self.sdn_switches) > 0:
            self.last_upgraded_switch = self.sdn_switches.keys().pop()
        else:
            self.last_upgraded_switch = None

    def ports_on_sdn_switches(self):
        """Return all the ports on sdn switches"""
        ports = []
        for switch in self.sdn_switches:
            ports.extend(self.topo.ports_on_switch(switch))

        return ports

    def collect_results(self):
        """Collect all the results from an iteration and append it to
        self.results"""
        topo = self.topo
        args = self.args

        # Total volume of traffic from all sdn ports
        sdnvol = sum((port.volume() for port in self.sdn_ports))
        totalvolume = sum((port.volume() for port in topo.ports()))

        # Max number of used vlans in any connected component
        max_used_vlans = max([len(used_vlans) for used_vlans in
                self.used_vlans_per_switch.itervalues()])

        # Max number of ft entries used on any SDN switch
        max_used_ft = args.maxft - min(self.remaining_ft_per_switch.values())
        totalftcapacity = args.maxft * len(self.sdn_switches)
        remainingftcapacity = sum(self.remaining_ft_per_switch.values())

        stretch_list = []
        for src_port in topo.ports():
            for dst_port in self.sim.communication_partners(src_port):
                newpath = []
                if src_port in self.sdn_ports:
                    newpath = src_port.path_to_port[dst_port]
                elif dst_port in self.sdn_ports:
                    # since we're only comparing length, we ignore that this is
                    # actually the reverse path
                    newpath = dst_port.path_to_port[src_port]
                else:
                    # both src and dst are not sdn ports and their paths have
                    # not changed
                    stretch_list.append(1)
                    continue
                dstswitch = dst_port.switch
                srcswitch = src_port.switch
                # we define path length as the len of the list of nodes
                orig_plen = len(topo.paths[srcswitch][dstswitch])
                new_plen = len(newpath)
                stretch_list.append(float(new_plen)/orig_plen)

        stretch_results = dict(
            avg_stretch2 = float(sum(stretch_list))/len(stretch_list),
            max_stretch2 = max(stretch_list),
            q25_stretch2 = np.percentile(stretch_list, 25),
            median_stretch2 = np.percentile(stretch_list, 50),
            q75_stretch2 = np.percentile(stretch_list, 75),
            q90_stretch2 = np.percentile(stretch_list, 90),
            q95_stretch2 = np.percentile(stretch_list, 95),
            q99_stretch2 = np.percentile(stretch_list, 99)
        )


        # reset load of the current solution
        link_utils = []
        link_relative_change_list = []
        link_absolute_change_list = []
        for n, nbrs in topo.graph.adjacency_iter():
            for nbr, eattr in nbrs.items():
                load = 8.0 * eattr['load-solution'] / 1000000
                load = load / topo.trace_duration
                util = load / eattr['link_capacity']
                link_utils.append(util)
                original_load = eattr.get('load', 0)
                if original_load != 0:
                    change = eattr['load-solution'] / original_load
                    link_relative_change_list.append(change)
                link_absolute_change_list.append(util)

        max_util = max(link_utils)

        link_results = dict(
            avg_link_relative_change = sum(link_relative_change_list)*1.0/len(link_relative_change_list),
            max_link_relative_change = max(link_relative_change_list),
            q25_link_relative_change = np.percentile(link_relative_change_list, 25),
            median_link_relative_change = np.percentile(link_relative_change_list, 50),
            q75_link_relative_change = np.percentile(link_relative_change_list, 75),
            q90_link_relative_change = np.percentile(link_relative_change_list, 90),
            q95_link_relative_change = np.percentile(link_relative_change_list, 95),
            q99_link_relative_change = np.percentile(link_relative_change_list, 99),
            avg_link_absolute_change = sum(link_absolute_change_list)*1.0/len(link_absolute_change_list),
            max_link_absolute_change = max(link_absolute_change_list),
            q25_link_absolute_change = np.percentile(link_absolute_change_list, 25),
            median_link_absolute_change = np.percentile(link_absolute_change_list, 50),
            q75_link_absolute_change = np.percentile(link_absolute_change_list, 75),
            q90_link_absolute_change = np.percentile(link_absolute_change_list, 90),
            q95_link_absolute_change = np.percentile(link_absolute_change_list, 95),
            q99_link_absolute_change = np.percentile(link_absolute_change_list, 99)
        )
        mlac = link_results['max_link_absolute_change']
        alac = link_results['avg_link_absolute_change']
        if len(self.sdn_ports) > 0:
            assert mlac >= alac, str((mlac, alac, self.args))
            if mlac == alac: logger.warn(str((mlac, alac, self.args)))
        mlrc = link_results['max_link_relative_change']
        alrc = link_results['avg_link_relative_change']
        if len(self.sdn_ports) > 0:
            assert mlrc >= alrc, str((mlrc, alrc, self.args))
            if mlrc == alrc: logger.warn(str((mlrc, alrc, self.args)))


        # Reformat the keys in the args dict so that they appear prefixed with
        # "param_" in the output of the simulation
        paramsdict = {}
        for k,v in args.__dict__.iteritems():
            paramsdict["param_"+str(k)] = v

        if len(self.sdn_ports) > 0:
            avg_pct_active_frontier = sum(
                [float(len(self.active_frontiers[i]))/len(self.frontiers[i])
                        for i in self.sdn_ports]
                ) / len(self.sdn_ports)
        else:
            avg_pct_active_frontier = 0

        result = dict(
                avg_pct_active_frontier=avg_pct_active_frontier,
                bestswitch=self.last_upgraded_switch,
                cost=0,
                #failedRebalancePorts=failedRebalancePorts,
                #feasiblePrioritySDNports=int(feasiblePrioritySDNports),
                frontierByComponent=self.frontier_by_component,
                iteration_duration=self.duration(),
                link_utils=link_utils,
                max_used_ft=max_used_ft,
                max_util=max_util,
                numcomponents=len(self.frontier_by_component.keys()),
                max_used_vlans=max_used_vlans,
                port_failure_reason=self.failure_reason,
                pct_ftentries=float(max_used_ft)/args.maxft,
                pct_sdnports=float(len(self.sdn_ports))/len(topo.ports()),
                pct_sdnvol=float(sdnvol)/totalvolume,
                pct_total_used_ftentries=float(totalftcapacity-remainingftcapacity)/totalftcapacity,
                pct_upgrades=float(len(self.sdn_switches))/topo.pretrimnodecount,
                pct_vlans=float(max_used_vlans)/args.maxvlans,
                #satisfiedBestEffortSDNports=satisfiedBestEffortSDNports,
                #satisfiedPrioritySDNports=satisfiedPrioritySDNports,
                #satisfiedSDNportsOnSDNswitches=satisfiedSDNportsOnSDNsws,
                sdnports=len(self.sdn_ports),
                sdnvol=sdnvol,
                total_used_ftentries=totalftcapacity-remainingftcapacity,
                tm_scaling_factor=topo.tm_scaling_factor,
                upgrades=len(self.sdn_switches)
                )
        result.update(paramsdict)
        result.update(link_results)
        result.update(stretch_results)

        #logger.info("{n} upgrades".format(n=result.get('upgrades')))
        logger.debug(
            ", ".join(["{k}={v}".format(k=k,v=v)
                for k,v in result.iteritems()
                if k != 'allpathstretch'])
        )
        self.results.append(result)


    def dump_results(self, checkpoint=False):
        args = self.args

        if not args.outputdir.endswith("/"):
            args.outputdir = args.outputdir + "/"
        filename = args.outputdir + ".".join([args.hostname, args.pid, args.starttime])

        if checkpoint:
            filename = "/data/dan.levin/" + ".".join([args.hostname, args.pid, args.starttime])
            filename += ".partial.json"
            try:
                file = gzip.open(filename, 'w')
                json.dump(self.results, file, indent=2)
                file.close()
            except:
                pass
        else:
            filename += ".final.json"
            file = gzip.open(filename, 'w')
            json.dump(self.results, file, indent=2)
            file.close()


    def iterations_so_far(self):
        """We define the number of iterations as the number of SDN switches
        that eave been upgraded so far"""
        return len(self.sdn_switches)

    def duration(self):
        return int(time() - self.starttime)

class Simulator(object):

    def __init__(self, topo, args):
        self.topo = topo
        self.graph = topo.graph
        self.args = args
        self.istate = IterationState(self)
        self.starttime = None

    def run(self):
        """ Minimize the stretch of the resulting topology subject to the
        constraint that the sum of switch upgrade costs are <= budget.  Returns
        'results' which is a list of dictionaries, each dictionary contains the
        results for an iteration of the switch upgrade loop
        """

        logger.info("Running Simulator")
        self.starttime = time()
        istate = self.istate

        # Ensure that every load-solution starts out as the load
        for src, dst, eattrs in self.topo.graph.edges(data=True):
            assert eattrs.get('load-solution') == \
                   eattrs.get('load'), str(eattrs)

        # Ensure path symmetry
        for src in self.topo.graph.nodes():
            for dst in self.topo.graph.nodes():
                assert self.topo.paths[src][dst] == \
                       list(reversed(self.topo.paths[dst][src]))

        #######################################################################
        # Iterate over every upgradable switch, according to the strategy
        #######################################################################
        strategy = self.args.switchstrategy

        while (self.switch_selection(istate, strategy) > 0) and \
            istate.iterations_so_far() <= self.args.toupgrade:

            logger.info("Iteration [{}] Started [{}].".format(
                        istate.iterations_so_far(),
                        istate.last_upgraded_switch
                        )
                    )


            istate.reinitialize()
            istate.frontiers = self.find_frontiers(istate)

            ###################################################################
            # For every SDN switch, allocate ft for each port on the switch.
            # This is the minimum ft state required, to call a port a "port"
            ###################################################################
            self.allocate_ft_for_ports_on_sdn_switches(istate)
            if __debug__:
                logger.debug("Allocated Basic Forwarding:\n{}".format(
                    "\t\n".join([str((k,v)) for k,v in
                                istate.remaining_ft_per_switch.iteritems()])
                    )
                )

            ###################################################################
            # Try to make every port "sdn", according to argsportstrategy
            ###################################################################
            if self.args.portstrategy == "sdn-switches-first":
                if __debug__:
                    logger.debug("trying [{}] ports on sdn switches".format(
                        len(istate.ports_on_sdn_switches())))
                self.make_ports_sdn(istate,
                                    cand_ports=istate.ports_on_sdn_switches())

            # try to maintain all the remaining ports that were sdn in the
            # previous iteration as sdn in this iteration
            if __debug__:
                logger.debug("trying [{}] sdn ports of previous iteration".\
                             format(len(istate.previous_sdn_ports)))
            self.make_ports_sdn(istate, cand_ports=istate.previous_sdn_ports)


            # try to make the remaining ports SDN
            if __debug__:
                logger.debug("trying all remaining ports")
            self.make_ports_sdn(istate)


            # For any ports that could not be made SDN due to insufficient
            # VLANs, ft entries or link capacities, try again aggressively,
            # selecting alternate possibly not shortest paths.
            #logger.debug("retrying aggressive all remaining sdn ports")
            #self.make_ports_sdn(istate, aggressive=True)

            ###################################################################
            # Assert the following invariants
            ###################################################################
            for port in self.topo.ports():
                if port not in istate.sdn_ports:
                    assert len(port.path_to_port) == 0, port.path_to_port

            for port in istate.sdn_ports:
                assert len(port.path_to_port) > 0, (sorted(istate.sdn_ports))

            numports = len(self.topo.ports())
            assert sum([int(len(rules) > numports) for rules in
                istate.basic_forwarding_rules_at_sw.values()]) == 0, \
                    "Some switch has more basic forwarding than allowed " + \
                    "[{}]".format(istate.basic_forwarding_rules_at_sw)

            istate.collect_results()
            istate.dump_results(checkpoint=True)
            logger.info("Iteration [{}] done. [{}/{}] SDN Ports in [{}/{}] secs".\
                format(len(istate.results),
                       len(istate.sdn_ports),
                       len(self.topo.ports()),
                       istate.duration(),
                       self.duration()))

        istate.dump_results()
        logger.info("Finished simulation.run")

################################################################################

    def duration(self):
        return int(time() - self.starttime)

    def allocate_ft_for_ports_on_sdn_switches(self, istate):
        """Ensure that every port on an SDN switch has bare minimum flow table
        state as determined by the traffic matrix, as well as for the
        guaranteed reachability between all ports"""

        for switch in istate.sdn_switches:
            comm_partners = []

            # Each unique destination port requires one ft entry.
            [comm_partners.extend(self.communication_partners(port)) for port in
                self.topo.ports_on_switch(switch)]
            comm_partners = set(comm_partners)

            # Also consider all ports on switch. Each one requires one ft entry.
            comm_partners = comm_partners.union(self.topo.ports_on_switch(switch))

            # One rule must exist for each comm_partner
            ft_entries = 0
            for port in comm_partners:
                ft_entries += istate.allocate_basic_forwarding_at_sw(switch,
                                                                     port)

            if __debug__:
                logger.debug("Consuming {} ft entries for switch {}".\
                             format(ft_entries,switch))

    def communication_partners(self, port):
        """A list of the ports with whom port communicates"""
        return [self.topo.port[port_name] for port_name in port.destinations()]

    def switch_selection(self, istate, strategy):
        """Determine the next switch in self.graph to designate as sdn
        according to strategy"""
        excluded = self.args.exclude

        if len(self.args.onlyupgrade) > 0:
            switches = [sw for sw in self.args.onlyupgrade
                        if sw not in istate.sdn_switches]

            # IF we're doing magic switches, upgrade them all at once
            if self.args.magicswitches > 0:
                for sw in switches:
                    istate.upgrade_switch(sw)
            elif len(switches) > 0:
                switch = switches[0]
                istate.upgrade_switch(switch)

            return len(switches)

        if strategy == "DEG":
            switches = [(d,n) for n,d in
                        nx.degree(self.graph).iteritems()
                        if n not in istate.sdn_switches and n not in excluded]

            if len(switches) > 0:
                value, switch = max(switches)
                istate.upgrade_switch(switch)

            return len(switches)

        if strategy == "VOL":
            switches = [(self.topo.egress_volumes_at_node(n),n) for n in
                        self.graph.nodes()
                        if n not in istate.sdn_switches and n not in excluded]

            if len(switches) > 0:
                value, switch = max(switches)
                istate.upgrade_switch(switch)

            return len(switches)

        if strategy == "RAND":
            switches = [n for n in self.graph.nodes()
                        if n not in istate.sdn_switches and n not in excluded]

            if len(switches) > 0:
                switch = random.choice(switches)
                istate.upgrade_switch(switch)

            return len(switches)


        if strategy == "ROUTERS":
            logger.debug("Upgrading all the Routers in one go")
            switches = [n for n in istate.topo.routers()
                        if n not in istate.sdn_switches]

            for switch in switches:
                istate.upgrade_switch(switch)

            return len(switches)

    def make_ports_sdn(self, istate, cand_ports=None, aggressive=False):
        """For every port candidate in cand_ports, ensure that it can be
        accommodated given the available vlans, ft entries, and link capacity to
        each destination.
        """

        topo = self.topo

        if cand_ports is None:
            cand_ports = sorted(list(set(topo.ports()) - istate.sdn_ports))
        else:
            cand_ports = sorted(list(set(cand_ports) - istate.sdn_ports))


        for candidate in sorted(list(cand_ports)):
            assert candidate not in istate.sdn_ports
            assert candidate.path_to_port == {}, "\n {} {}".format(candidate,
                    candidate.path_to_port)

            logger.debug("trying port [{}] sdn".format(candidate))
            self.make_port_sdn(istate, candidate, aggressive)

    def vlan_constraints_met(self, istate, candidate):
        active_frontier = self.find_active_frontier(istate, candidate)
        istate.active_frontiers[candidate] = active_frontier
        if len(active_frontier) == 0:
            istate.failure_reason.setdefault('no_vlan', 0)
            istate.failure_reason['no_vlan'] += 1
            return False
        return True

    def verify_frontier_path(self, candidate, dst_port, path):
        """Safety assertion that paths are correctly chosen"""

        path_to_frontiersw = candidate.path_to_frontiersw.\
                             get(candidate.designated_switch.get(dst_port))
        for mustbetrue in set([path[i] == path_to_frontiersw[i] for i,sw
            in enumerate(path_to_frontiersw)]):
            assert mustbetrue is True, "path different from \
                  frontierpath: [{}] -> [{}]\nsct: [{}]\nvia: [{}]\norg: [{}]".\
                format(
                    candidate, dst_port,
                    path_to_frontiersw,
                    path,
                    self.topo.paths[candidate.switch][dst_port.switch])

    def make_port_sdn(self, istate, candidate, aggressive):
        """Try to make the candidate port SDN"""

        topo = self.topo

        # 1 ensure that VLAN Constraints are met to each switch on the
        # frontier. This defines the active_frontier for candidate.
        if not self.vlan_constraints_met(istate, candidate):
            return

        # Record the path that the candidate takes to every port. should the
        # candidate port become SDN, this is how it forwards traffic to each
        # destination
        path_to_port = OrderedDict()
        mark_port_sdn = True

        # Keep the state of the graph prior to changing any ft entries or
        # link taffic projections for this port. If this candidate fails, we
        # will roll these back
        rollback_ft_allocations = copy.deepcopy(istate.remaining_ft_per_switch)
        # Not necessary to rollback basic_forwarding_rules_at_sw because
        # - Each port is only tried to be made SDN once per simulation iteration
        # - The remaining ft entries are reset
        # - Before each iteration, the forwarding rules on the SDN switches are
        #   reset using IterationState.reinitialize
        rollback_graph = {}
        for src, dst, edgeattrs in topo.graph.edges(data=True):
            rollback_graph[(src, dst, 'load-solution')] = \
                                                edgeattrs['load-solution']
            #rollback_graph[(src, dst, 'flows')] = edgeattrs['flows'].copy()

        # NOTE: Everything inside here must be FAST. This is the inner loop
        # check_link capacity only needs to handle candidate.destinations().
        # candidate.desinations() just returns the ports to which >0 bytes are
        # projected.
        for dst_port_name in candidate.destinations():
            dst_port = topo.port[dst_port_name]
            #NOTE: safe_paths is where we would be aggressive (see method doc
            #      for details)
            path = self.safe_paths(istate, candidate, dst_port)
            self.verify_frontier_path(candidate, dst_port, path)

            # 2 ensure that FT constraints are met along the path.  Check
            # that there exists args.epp ft emtries available at
            # designed_sdn_switch(port). We might be able to optimize this
            # as long as the sum of free ft entries over all the sdn
            # switches between port -> dst. consume ft entries according to
            # any of the three possible ISF strategies

            if not self.check_sufficient_ft(istate,
                                            candidate,
                                            dst_port,
                                            path):
                # If we are here, we failed to find sufficent ft capacity
                istate.failure_reason.setdefault('link_ft', 0)
                istate.failure_reason['link_ft'] += 1
                mark_port_sdn = False
                break


            # 3 ensure that traffic constraints are met
            # Check that link capacity exists for the traffic demand in both
            # directions from port -> dst.
            # Would be better to run a list of overloaded links... if
            # len(overloaded_links) > 0:
            if self.check_sufficient_link_cap(istate,
                                              candidate,
                                              dst_port,
                                              path) is False:
                istate.failure_reason.setdefault('link_cap', 0)
                istate.failure_reason['link_cap'] += 1
                mark_port_sdn = False
                break

            # Keep the path that candidate would use to reach this dst_port
            # IF candidate becomes sdn
            path_to_port[dst_port] = path

        if mark_port_sdn:
            logger.debug("Made port [{}] sdn:".format(candidate))
            istate.sdn_ports.add(candidate)
            candidate.path_to_port = path_to_port

            # Update the reverse path of all other SDN switches that already
            # had a path to candidate. This update is necessary, as
            # candidate now takes a path over both its designated switch and
            # dst_port's designated switch.
            for port in istate.sdn_ports:
                if port is not candidate:
                    fwd_path = candidate.path_to_port.get(port)
                    if fwd_path is not None:
                        port.path_to_port[candidate] = \
                            list(reversed(fwd_path))
        else:
            #roll back all ft allocations and traffic
            self.roll_back_ft_allocations_traffic(rollback_ft_allocations,
                                                      rollback_graph)

    def roll_back_ft_allocations_traffic(self,
                                         rollback_ft_allocations,
                                         rollback_graph):
        self.istate.remaining_ft_per_switch = rollback_ft_allocations
        for src, dst, edgeattrs in self.topo.graph.edges(data=True):
            #for i in ['flows', 'load-solution']:
            for i in ['load-solution']:
                edgeattrs[i] = rollback_graph[(src, dst, i)]

    def safe_paths(self, istate, src_port, dst_port):
        """Returns a list of "safe" paths from src_port's node to
        dst_port's node which traverse at least one SDN switch.

        if dst is not sdn port, then a valid path is a path which include at
        least one sdn switch on the topological frontier of src_port. This sdn
        switch becomes the designated switch for src_port to reach dst_port.

        if dst is sdn port, then a valid path is a path which includes at
        least one sdn switch on the topological frontier of src_port, and the
        designated switch of dst_port.

        The returned path is always a shortest path. In the future, non
        shortest paths could be chosen if the src_port can not be made SDNc
        using a shortest path. The aggressive parameter is intended for that.
        """
        topo = self.topo
        best_path = None

        if dst_port not in istate.sdn_ports:
            for frontier_sw in istate.active_frontiers[src_port]:
                path = src_port.path_to_frontiersw[frontier_sw][:-1]
                rest = topo.paths[frontier_sw][dst_port.switch]
                path.extend(rest)

                if best_path is None or \
                   (path is not None and len(path) < len(best_path)):
                    best_path = path
                    src_port.designated_switch[dst_port] = frontier_sw

            return best_path

        else:
            # dst_port is an SDN port and any valid path from src_port to
            # dst_port must traverse the designated switch of dst_port
            dst_designated_switch = dst_port.designated_switch[src_port]

            # append to the suffix of the path, the previously determined path
            # between dst_port and its designated switch for src_port (but in
            # reverse)
            dst_sct_path = list(
                            reversed(dst_port.\
                                     path_to_frontiersw[dst_designated_switch]))

            for frontier_sw in istate.active_frontiers[src_port]:
                # NOTE: enhancement: model the 3 different types of ISF in
                # shortes_path_via path_one =
                # topo.shortest_path_via(src_port.switch,
                #                   dst_port.switch,
                #                   [frontier_sw, dst_designated_switch])

                isfpath = topo.paths[frontier_sw][dst_designated_switch]
                path = src_port.path_to_frontiersw[frontier_sw][:-1]
                path.extend(isfpath[:-1])
                path.extend(dst_sct_path)

                if best_path is None or \
                   (path is not None and len(path) < len(best_path)):
                    best_path = path
                    src_port.designated_switch[dst_port] = frontier_sw

            return best_path

    def find_active_frontier(self, istate, port):
        """Returns a list of switches that make up the active frontier of
        port. The active frontier is the (sub)set of frontier switches for which
        sufficient VLANs permit each switch to be used as a designated sdn
        switch toward some destination.

        restrict_size may be set to an int N, which limits the active frontier to
        at most N switches
        """
        topo = self.topo
        possible_legacy_switches = set()
        committed_legacy_switches = set()
        active_frontier = []
        vlan_id = None

        # sort the switches on frontier of port by distance from port
        sorted_frontier = sorted(
            ((len(topo.paths[port.switch][sw]), sw)
             for sw in istate.frontiers[port])
        )

        # Limit the number of switches on the active frontier
        if self.args.activefrontiersize is not None:
            restrict_size = self.args.activefrontiersize
            assert restrict_size > 0
            sorted_frontier = sorted_frontier[:restrict_size]

        # Try every frontier switch as a candidate for the active_frontier
        for distance, frontier_switch in sorted_frontier:
            path = nx.shortest_path(topo.graph, port.switch, frontier_switch)

            sdnswitches = [sw for sw in path[:-1] if sw in istate.sdn_switches]

            # If any sdn switches are found in the path, we use a copygraph to
            # avoid the situation where multiple shortest paths exist to a
            # frontier switch, each of which may traverse other frontier
            # switches. We remove every other frontier_switch to prevent
            # traversing the wrong path.
            if len(sdnswitches) != 0:
                tmpfrontier = set(istate.frontiers[port])
                tmpfrontier.remove(frontier_switch)
                copygraph = nx.Graph(topo.graph)
                copygraph.remove_nodes_from(tmpfrontier)
                newpath = nx.shortest_path(copygraph, port.switch, frontier_switch)
                #logger.info("""weird situation:
                #sdnswitches {}
                #invalid sct {}
                #valid sct   {}""".format(sdnswitches, list(path), list(newpath)))
                path = newpath

            # Collect the legacy switches on the shortest path to the frontier
            # and skip the last switch (the frontier switch at the end)
            for legacy_switch in path[:-1]:
                possible_legacy_switches.add(legacy_switch)

            sdnswitches = [sw for sw in possible_legacy_switches if sw in
                    istate.sdn_switches]

            assert len(sdnswitches) == 0, \
                """NOW WE REALLY HAVE A PROBLEM. No switches should be sdn here but {} are.
                path     {}
                lgcy sw  {}
                port     {}
                frontier {}""".format(sdnswitches,path,possible_legacy_switches,
                        port,istate.frontiers[port])

            # Record the path taken from the port to its frontier switch
            port.path_to_frontiersw[frontier_switch] = path

            # Try to find a common vlan ID among all legacy switches encountered
            # thus far.
            candidate_vlan_id = self.find_common_vlan(istate, possible_legacy_switches)
            if candidate_vlan_id is not None:
                vlan_id = candidate_vlan_id
                active_frontier.append(frontier_switch)
                committed_legacy_switches = possible_legacy_switches

        # allocate the vlan id to each legacy switch
        if vlan_id is not None:
            for switch in committed_legacy_switches:
                istate.used_vlans_per_switch[switch].add(vlan_id)
                logger.debug("Allocating vlan [{}] on [{}]".format(
                    istate.used_vlans_per_switch[switch], switch)
                )

        logger.debug("Active Frontier of [{}] is [{}]".format(port, active_frontier))
        return active_frontier

    def find_common_vlan(self, istate, switches):
        """Returns the lowest free vlan id from a set of legacy switches or None
        if there is no vlan id is common among them"""


        used_vlans = set()
        max_usable_vlans = self.args.maxvlans

        for switch in switches:
            used_vlan_ids = istate.used_vlans_per_switch.get(switch, [])

            # we can not allocate another vlan on switch
            if len(used_vlan_ids) >= max_usable_vlans:
                logger.debug("No more vlans on {}".format(switch))
                return None

            used_vlans = used_vlans.union(used_vlan_ids)

        # Compute all of the vlans common to all of the legacy switches
        available_vlans = set(range(max_usable_vlans)).difference(used_vlans)
        if len(available_vlans) == 0:
            return None
        else:
            return min(available_vlans)

    def find_frontiers(self, istate):
        """Return a dictionary of frontier by port"""

        graph = self.graph
        frontiers = {}

        component_by_id = {}
        component_id_by_port = {}
        frontier_by_component_id = {}

        undirected_graph = nx.Graph(graph)
        undirected_graph.remove_nodes_from(istate.sdn_switches)
        components = nx.connected_components(undirected_graph)

        # Determine which component every port belongs to
        for cid, component in enumerate(components):
            component_by_id[cid] = component
            for cnode in component:
                ports = istate.topo.ports_on_switch(cnode)
                for port in ports:
                    component_id_by_port[port] = cid

        # Determine which sdn_switches belong to each component
        for cid, component in component_by_id.iteritems():
            frontier = set([])
            for switch in component:
                for i in nx.neighbors(graph, switch):
                    if i in istate.sdn_switches:
                        frontier.add(i)

            frontier_by_component_id[cid] = list(frontier)
            #logger.debug("frontier of cid {} is {}".format(cid, list(frontier)))

        # SDN ports on an SDN have the frontier of JUST the SDN switch
        for switch in istate.sdn_switches:
            ports_on_sdnswitch = istate.topo.ports_on_switch(switch)
            for port in ports_on_sdnswitch:
                frontiers[port] = [switch]

        # if a port is not SDN, then its frontier includes all SDN switches
        for port in istate.topo.ports():
            # if port is in frontiers, it means it was on an SDN switch
            if port in frontiers:
                continue
            frontiers[port] = frontier_by_component_id[component_id_by_port[port]]

        istate.frontier_by_component = frontier_by_component_id

        return frontiers

    def check_sufficient_ft(self, istate, candidate, dst_port, path):
        """Check that every sdn switch along path has sufficient ft entries to
        accommodate the candidate port as an sdn port

        If so, return True
        If we can't allocate ft entries across the SDN switches, return False"""

        required_rules = self.args.epp
        sdn_switches = [sw for sw in path if sw in istate.sdn_switches]

        # Every sdn switch must have one ft entry for both the candidate and
        # dst_port to enable basic IP forwarding
        basic_rules = 0
        for switch in sdn_switches:
            for port in [dst_port, candidate]:
                basic_rules += istate.allocate_basic_forwarding_at_sw(switch,
                                                                      port)
                logger.debug("check_ft allocated Basic Forwarding: {}".format(
                    str(basic_rules)))

        # Try to greedily accomodate required_rules for the policy on each
        # switch in order of distance from the candidate.
        remaining_rules = required_rules
        for switch in sdn_switches:
            rules = min(istate.remaining_ft_per_switch[switch], remaining_rules)
            istate.remaining_ft_per_switch[switch] -=  rules
            remaining_rules -= rules
            if remaining_rules == 0:
                return True

        # We could not allocate sufficient ft entries for candidate
        # We rely upon the make_ports_sdn method to revert the
        # remaining_ft_per_switch and _allocate_basic_forwarding_at_sw in case
        # we return False
        return False


    def check_sufficient_link_cap(self, istate, candidate, dst_port, path):
        """Check that every link along the path can accomodate the traffic
        demand in BOTH directions. If so, then allocate it atomically.
        """
        topo = self.topo

        src_switch = path[0]
        dst_switch = path[-1]

        forward_traffic = candidate.volume_to(dst_port)
        reverse_traffic = dst_port.volume_to(candidate)

        forward_path = path
        reverse_path = list(reversed(path))

        # if dst_port is SDN we can't use topo.paths because dst_port already
        # uses a different path to reach candidate (and vice versa)
        if dst_port in istate.sdn_ports:
            reverse_original_path = dst_port.path_to_port[candidate]
            forward_original_path = list(reversed(reverse_original_path))

            if forward_path == forward_original_path:
                assert reverse_path == reverse_original_path, (reverse_path,
                        reverse_original_path)
                return True
            elif reverse_path == reverse_original_path:
                assert forward_path == forward_original_path, (forward_path,
                        forward_original_path)
                return True


        else:
            forward_original_path = topo.paths[src_switch][dst_switch]
            reverse_original_path = topo.paths[dst_switch][src_switch]
            assert forward_original_path == list(reversed(reverse_original_path)), (forward_original_path, reverse_original_path)

            if forward_path == forward_original_path:
                assert reverse_path == list(reversed(forward_path))
                assert reverse_path == list(reversed(forward_original_path))
                assert reverse_path == reverse_original_path, (reverse_path, reverse_original_path)
                return True
            elif reverse_path == reverse_original_path:
                assert forward_path == forward_original_path, (forward_path,
                        forward_original_path)
                return True

        # Set the argparsed link utilization threshhold which is not to be
        # exceeded during projection.
        thresh = self.args.linkutilthresh

        # Note that topo.project_load_on_path is assumed to be atomic. It
        # returns False with not changes to any link load, or it projects the
        # traffic and returns True.
        # 1) subtract traffic from the original path (in both directions!)
        assert len([sw for sw in path if sw in istate.sdn_switches]) > 0
        topo.project_load_on_path(forward_original_path, -forward_traffic,
                thresh, candidate, dst_port)
        topo.project_load_on_path(reverse_original_path, -reverse_traffic,
                thresh, dst_port, candidate)


        # 2) Check that we can allocate the traffic over path
        if topo.project_load_on_path(forward_path, forward_traffic, thresh, candidate, dst_port):
            if topo.project_load_on_path(reverse_path, reverse_traffic, thresh, dst_port, candidate):
                return True
            else:
                topo.project_load_on_path(forward_path, -forward_traffic,
                        thresh, candidate, dst_port)

        # Replace the traffic for the original paths
        topo.project_load_on_path(forward_original_path, forward_traffic,
                thresh, candidate, dst_port)
        topo.project_load_on_path(reverse_original_path, reverse_traffic,
                thresh, dst_port, candidate)
        return False

    def ports_per_component(self, istate):
        """Returns a dict of number of ports per topological component"""

        graph = self.graph

        maxports = istate.iterations_so_far()

        undirected_graph = nx.Graph(graph)
        undirected_graph.remove_nodes_from(istate.sdn_switches)
        components = nx.connected_components(undirected_graph)

        # Determine which component every port belongs to
        ports_in_component = {}
        for cid, component in enumerate(components):
            ports_in_component[cid] = 0
            for cnode in component:
                ports_in_component[cid] += \
                    len(istate.topo.ports_on_switch(cnode))

        return ports_in_component

    def debug_loadproj(self):
        """confirm that the original paths over which load is projected are
        correct"""
        topo = self.topo
        for candidate in topo.ports():
            for dst_port in candidate.destinations():
                dst_port = topo.port[dst_port]
                forward_traffic = candidate.volume_to(dst_port)
                reverse_traffic = dst_port.volume_to(candidate)

                src_switch = candidate.switch
                dst_switch = dst_port.switch

                forward_original_path = topo.paths[src_switch][dst_switch]
                reverse_original_path = topo.paths[dst_switch][src_switch]

                topo.project_load_on_path(forward_original_path, -forward_traffic, candidate, dst_port)
                #topo.project_load_on_path(reverse_original_path, -reverse_traffic, dst_port, candidate)
