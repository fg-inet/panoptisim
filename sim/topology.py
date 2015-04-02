# Dan Levin <dan@net.t-labs.tu-berlin.de>

import clusters
from collections import OrderedDict
import json
import logging
import logging.config
import networkx as nx
import numpy
import random
import os
import pickle
import gzip

from sim.enterprisedata import EnterpriseData
from traces.traces import trace_durations

FORMAT="%(asctime)s %(process)d %(name)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

class Port(object):
    """keeps topological and volume state for an endpoint"""

    def __init__(self, myname, switch):
        self.name = str(myname)
        self.switch = str(switch)
        self._volume_to = {}

        # Ephemeral state that depends on port assignment
        # It is expected that this is reset in IterationState
        self.designated_switch = OrderedDict()
        self.path_to_port = OrderedDict()
        self.path_to_frontiersw = OrderedDict()

    def __repr__(self):
        return "{}.{}".format(self.switch, self.name)

    def __lt__(self, other):
        return int(self.name) < int(other.name)

    def destinations(self):
        """Return the list of destinations to which traffic goes"""
        return self._volume_to.keys()

    def set_volume_to(self, dst_port, vol):
        """Set the traffic volume destined from this port to dst_port"""
        self._volume_to[dst_port.__repr__()] = vol

    def scale_traffic(self, factor):
        """Scale the traffic to all ports by factor"""
        for dst, vol in self._volume_to.iteritems():
            self._volume_to[dst] = vol * factor

    def volume_to(self, port):
        """Returns the volume of traffic to a port or another node. The
        volume_to a node will be the sum of the traffic to the ports of that
        node."""
        return self._volume_to.get(port.__repr__(), 0)

    def volume(self):
        """Returns the total outbound traffic from this port"""
        return sum(self._volume_to.values())


class EnterpriseTopology(object):
    """Enterprise Network Topology"""

    def __init__(self, costfun="nocore", tm="2004", tm_factor_name="max-50",
            EPscalingFactor=1, mappingseed=None):
        """ costfun is a string in ["10xcore", "nocore"] """

        logger.info("Initializing a new EnterpriseTopology")
        logger.debug("Traffic matrix: %s" % str(tm))
        logger.debug("Traffic costfun: %s" % str(costfun))

        # Use this to initialize the random seed when a EnterpriseTopology instance
        # is not created using panoptisim.py
        if mappingseed is not None:
            random.seed(mappingseed)

        self.name = "enterprise"
        self.tm = tm
        self.port = OrderedDict()

        endpoint_scaling_factor = EPscalingFactor
        assert EPscalingFactor == int(EPscalingFactor)

        self.trace_duration = trace_durations[tm]
        lbnl = json.load(open("traces/tm"+tm+".json"))

        self.graph = self.__enterprise_to_nx()
        self.leaves = self.__prune_access_switches(endpoint_scaling_factor)
        self.paths = nx.all_pairs_shortest_path(self.graph)

        #Ensure symmetry in the original shortest paths
        for src in self.graph.nodes():
            for dst in self.graph.nodes():
                self.paths[src][dst] = list(reversed(self.paths[dst][src]))

        # Map endpoints and project load over the shortest path topology
        self.__map_end_points(lbnl)
        self.__project_base_load()

        # Scale up the traffic according to the tm_factor_name and re-project
        # new scaled-up traffic load
        self.tm_scaling_factor = self.__calculate_traffic_scaling_factor(tm_factor_name)
        self.__scale_traffic(self.tm_scaling_factor)
        self.__project_base_load()

        #Ensure every link has at any flow traversing it at most once
        for src, dst, e in self.graph.edges(data=True):
            flows_by_srcdst = e['flows']
            for key, value in flows_by_srcdst.iteritems():
                assert value == 1


    @staticmethod
    def load_from_pickle(tm, tmsf, seedmapping, EPscalingFactor):
        """ Loads a pickled topology. Throws an exception when topo file is not
        found. """

        filename = "pickles/enterprise-tm"+\
                tm+\
                str(tmsf)+\
                str(seedmapping)+\
                str(EPscalingFactor)+".pickle"
        logger.info("Unpickling [{}]".format(filename))
        try:
            os.mkdir("pickles")
        except OSError:
            pass

        topo = None
        try:
            tf = gzip.open(filename, 'r')
            topo = pickle.load(tf)
            tf.close()
            logger.info("Unpickling Successful")
        except:
            logger.warn("Unpickling Failed, Creating New EnterpriseTopology")
            topo = EnterpriseTopology(tm=tm, tm_factor_name=tmsf,
                                  EPscalingFactor=EPscalingFactor)
            tf = gzip.open(filename, 'w')
            pickle.dump(topo, tf)
            tf.close()
            logger.info("Pickled EnterpriseTopology from: {}".format(filename))

        return topo


    def __enterprise_to_nx(self):
        """ Transform the Enterprise topology into a networkx graph. """

        links = EnterpriseData().links()

        graph = nx.DiGraph()
        for link in links:
            src = link['src']
            dst = link['dst']

            # Adding an edge with no attributes is expected to be idempotent
            graph.add_edge(src, dst)
            linkcap = graph.edge[src][dst].get('link_capacity', 0) + link['link_capacity']
            #name = link['portname']
            graph.add_edge(src, dst, link_capacity=linkcap)

            #graph.node[src]['portcount'] = 0
            graph.node[src]['is_router'] = link['is_router']
            graph.node[src]['endpointcount'] = 0
            graph.node[dst]['endpointcount'] = 0
            # Temporary state for determining how to assign ports
            graph.node[src]['rec_endpointcount'] = 0
            graph.node[dst]['rec_endpointcount'] = 0


        # Correct for edge asymmetry and set each link capacity to the max of
        # either direction

        bits_per_byte = 8
        bits_per_MBit = 1000000
        for edge in graph.edges():
            src, dst = edge

            # Add the reverse direction link if it doesn't exist
            if graph.edge.get(dst, {}).get(src) is None:
                graph.add_edge(dst, src)

            stod = graph.edge[src][dst].get('link_capacity', 0)
            dtos = graph.edge.get(dst, {}).get(src, {}).get('link_capacity', 0)
            cap = max(stod, dtos)
            graph.edge[src][dst]['link_capacity'] = cap
            graph.edge[dst][src]['link_capacity'] = cap
            # This value is the total number of bytes at the full link speed of the
            # link times the duration of the traffic trace
            graph.edge[src][dst]['load-limit'] = float(cap * bits_per_MBit) * self.trace_duration / bits_per_byte
            graph.edge[dst][src]['load-limit'] = float(cap * bits_per_MBit) * self.trace_duration / bits_per_byte

        # The enterprise topology is not connected!
        # So we will just take the largest connected component
        # We take the largest component, 1577
        largest_component = [ x for x in
                nx.connected_component_subgraphs(graph.to_undirected()) if
                len(x.nodes()) == 1577 ][0]

        newgraph = largest_component.to_directed()
        # Assert that every edge is the same
        for edge in newgraph.edges():
            src, dst = edge

            assert newgraph[src][dst] == graph[src][dst], \
                str((newgraph[src][dst], graph[src][dst]))
            assert newgraph[dst][src] == graph[dst][src], \
                str((newgraph[dst][src], graph[dst][src]))

        self.pretrimnodecount = len(graph.nodes())

        return newgraph

    def __prune_access_switches(self, endpoint_scaling_factor):
        """Identify and remove nodes in the graph which we consider to be access
        switches."""
        numallendpoints = 0

        # Note that leaves in a directed graph have node degree 2
        leaves = [ k for k,v in self.graph.degree().iteritems() if v == 2 ]
        for i in leaves:
            self.graph.node[i]['isaccess'] = True
            self.graph.node[self.graph.neighbors(i)[0]]['endpointcount'] += endpoint_scaling_factor
            numallendpoints += endpoint_scaling_factor

        #1293 was the equivalent number of access switches in the undirected graph
        #There are certain distribution switches that have an end-point that
        #does not come from having removed an access switch
        for i in [ k for k,v in self.graph.degree().iteritems() if v == 4 ]:
            if self.graph.node[i]['endpointcount'] == 0:
                self.graph.node[i]['endpointcount'] = endpoint_scaling_factor
                numallendpoints += endpoint_scaling_factor

        self.graph.remove_nodes_from(leaves)

        self.numallendpoints = numallendpoints

        #self.graph = nx.convert_node_labels_to_integers(self.graph, discard_old_labels=False)

        index = 0
        self.endpoint_to_node = {}
        for i in self.graph.nodes():
            prev_index = index
            n = self.graph.node[i]
            n['endpoints'] = range(index, index+n['endpointcount'])
            n['rec_endpointcount'] = n['endpointcount']
            for j in range(n['endpointcount']):
                self.endpoint_to_node[index] = i
                index += 1
            assert index == prev_index + n['endpointcount']


        return leaves


    def __map_end_points(self, tm):
        """Assign endpoints from the traffic matrix tm to nodes in the graph We
        take config1515 in the enterprise graph to be the upstream provider gateway
        based (arbitrarily) on its location and it having the highest
        degree-distribution in the network.
        """
        gatewaynode='config1515'
        #config1515 has 65 l2 adjacent neighbors before trimming (bi-directional degree of 130)
        #assert nx.degree(self.graph, gatewaynode) == 65*2, len(self.graph.edge['config1515'])
        #config1515 has 53 l2 adjacent neighbors after trimming leaves
        assert nx.degree(self.graph, gatewaynode) == 53*2, len(self.graph.edge['config1515'])

        clusters_to_ips = clusters.get_clusters(tm)
        numAllIPs = sum(map(len, clusters_to_ips.values()))
        ipsPerEndPoint = numAllIPs * 1.0 / self.numallendpoints
        assert ipsPerEndPoint >= 1.0

        logger.debug("# nodes " + str(len(self.graph.nodes())))
        logger.debug("# end-points " + str(self.numallendpoints))
        logger.debug("# all IPs " + str(numAllIPs))
        logger.debug("# IPs per end-point " + str(ipsPerEndPoint))

        cluster_to_results = {}
        endpoints = []

        num_clusters = len(clusters_to_ips.keys())

        # compute how many end points are needed for each cluster
        end_points_per_cluster = {}
        used_end_points = self.numallendpoints
        for key, ips in clusters_to_ips.iteritems():
            if key == 'gateway':
                continue
            numips = len(clusters_to_ips[key])
            how_many_end_points = int(numips / ipsPerEndPoint)
            end_points_per_cluster[key] = how_many_end_points
            used_end_points -= how_many_end_points

        # there are used_end_points remaining end points that need a cluster
        for j in range(used_end_points):
            k = j % num_clusters
            key = clusters_to_ips.keys()[k]
            end_points_per_cluster[key] += 1

        for key, ips in clusters_to_ips.iteritems():

            if key == 'gateway':
                continue

            #reset visited for each iteration
            for i in self.graph.nodes():
                self.graph.node[i]['visited'] = False

            how_many_end_points = end_points_per_cluster[key]

            logger.debug("Find end-points for cluster " + key)
            logger.debug("Looking for # end points " + str(how_many_end_points))
            # Our radius is defined "large enough" as 10* the diameter
            endpoint_nodes = self.__find_end_points(how_many_end_points, 10 * nx.diameter(self.graph))
            logger.debug("Found # end points " + str(len(endpoint_nodes)))
            assert how_many_end_points == len(endpoint_nodes)

            # map IPs to end-points
            ips_slices = {}
            for i in range(how_many_end_points):
                ips_slices[i] = []

            for j, ip in enumerate(ips):
                k = j % how_many_end_points
                ips_slices[k].append(ip)

            cluster_to_results[key] = (endpoint_nodes, ips_slices)
            for i in range(how_many_end_points):
                # endpoints is a list of lists
                endpoints.append((ips_slices[i],endpoint_nodes[i]))

        # assert we have clustered all end points
        for i in self.graph.nodes():
            n = self.graph.node[i]
            assert len(n['endpoints']) == 0
            assert n['rec_endpointcount'] == 0
            n.pop('endpoints')
            n.pop('rec_endpointcount')

        gw = self.graph.node[gatewaynode]
        gw['endpointcount'] += 1
        self.numallendpoints += 1
        gwid = len(self.endpoint_to_node.keys())
        self.endpoint_to_node[gwid] = gatewaynode
        #logger.debug "gatewayid " + str(gwid)
        endpoints.append((['gateway'], gwid))

        # Record the ip addresses belonging to each endpoint
        self.ips_by_id = {}
        for endpoint_tuple in endpoints:
            endpoint_ip_list = endpoint_tuple[0]
            endpoint_id = endpoint_tuple[1]
            self.ips_by_id[endpoint_id] = endpoint_ip_list

        for j, ep1pair in enumerate(endpoints):
            for k, ep2pair in enumerate(endpoints):
                if j == k:
                    continue
                vol = 0

                # each endpoint is made up of multiple ip addresses from the
                # lbnl trace. we aggregate traffic over all ips per endpoint
                for ip1 in ep1pair[0]:
                    if ip1 in tm:
                        source = tm[ip1]
                        for ip2 in ep2pair[0]:
                            if ip2 in source:
                                vol += source[ip2]

                if vol > 0:
                    ep1idx = ep1pair[1]
                    ep2idx = ep2pair[1]
                    src_switch = self.endpoint_to_node[ep1idx]
                    dst_switch = self.endpoint_to_node[ep2idx]
                    src_node = self.graph.node[src_switch]
                    dst_node = self.graph.node[dst_switch]
                    src_port = src_node.setdefault('ports', {}).get(ep1idx)
                    dst_port = dst_node.setdefault('ports', {}).get(ep2idx)

                    if src_port is None:
                        #src_port = Port(ep1idx, src_switch, src_node)
                        src_port = Port(ep1idx, src_switch)
                        src_node['ports'][ep1idx] = src_port
                        self.port[str(src_port)] = src_port

                    if dst_port is None:
                        #dst_port = Port(ep2idx, dst_switch, dst_node)
                        dst_port = Port(ep2idx, dst_switch)
                        dst_node['ports'][ep2idx] = dst_port
                        self.port[str(dst_port)] = dst_port

                    src_port.set_volume_to(dst_port, vol)

    def reinitialize(self):
        self.__project_base_load()
        # Some edges which had no load before will get load-solution
        for src, dst, edge_attrs in self.graph.edges(data=True):
            edge_attrs['load-solution'] = edge_attrs.get('load', 0)

    def endpoint_volume_from_node(self, node):
        """Returns the a list of traffic volume in bytes from by the endpoints
        of node toward to every other node in the graph

        [(55791682.63987819, 'config1565'),]

        Means that the endpoints at 'node' generate 55791682.63987819 bytes
        toward 'config1565'.
        """
        vol_per_endpoint = [sum(port._volume_to.values())
            for port in self.graph.node[node].get('ports', {}).values()]
        return sum(vol_per_endpoint)

    def endpoint_volume_to_node(self, node):
        """Returns the traffic volume in bytes from the endpoints
        of node toward every other node in the graph
        """
        volume = 0

        dstports = self.ports_on_switch(node)
        for dstport in dstports:
            for port in self.ports():
                volume += port.volume_to(dstport)

        return volume

    def egress_volumes_at_node(self, node):
        """Returns the a list of traffic volume in bytes egressing from node
        toward each adjacent neighbor node"""

        return sorted([(attr.get('load-solution', 0), dst)
            for dst, attr in self.graph.edge[node].iteritems()
            ])

    def ingress_volumes_at_node(self, node):
        """Returns the a list of traffic volume in bytes egressing from node
        toward each adjacent neighbor node"""
        volume = []
        for neighbor in nx.neighbors(self.graph, node):
            invol = self.graph.edge[neighbor][node].get('load-solution', 0)
            volume.append((invol, neighbor))

        return sorted(volume)

    def ports_on_switch(self, node):
        """Returns the port instances on the graph node"""
        return self.graph.node[node].get('ports', OrderedDict()).values()

    def __calculate_traffic_scaling_factor(self, tm_factor_name):
        """Takes an lbnl_year (2004 or 2005) and returns a dict of traffic
        matrix scaling factors"""

        factors = {'max-50': 0,
                   '0.5-50': 0,
                   '1-50': 0,
                   '1': 1
                  }

        assert factors.get(tm_factor_name, None) is not None, "Invalid tm scaling factor name"

        # calculate the link utilization in terms of percentage of link capacity
        # load is in bytes, we convert everything to bits/second.
        # for every link

        raw_utils = self.sorted_linkutil(data=True)
        acceptable_scaling_factor = None

        util_classes = []

        utils_all   = [util for util, src, dst, linkcap in raw_utils]
        utils_100   = [util for util, src, dst, linkcap in raw_utils if linkcap == 100]
        utils_1000  = [util for util, src, dst, linkcap in raw_utils if linkcap == 1000]
        utils_2000  = [util for util, src, dst, linkcap in raw_utils if linkcap == 2000]
        utils_10000 = [util for util, src, dst, linkcap in raw_utils if linkcap == 10000]

        util_classes = [utils_100, utils_1000, utils_2000, utils_10000]

        link_cap_scaling_factors = []
        # Find scaling factor for each class of link capacity
        for utils in util_classes:
            # scale up every src,dst demand so that the most utilized link in the
            # network is 50% utilized
            factors['max-50'] = 0.50/(max(utils))
            # .. so that the 99.5th percentile utilized link is 50% utilized
            factors['0.5-50'] = 0.5/numpy.percentile(utils, 99.5)
            # .. so that the 99th percentile utilized link is 50% utilized
            factors['1-50'] = 0.5/numpy.percentile(utils, 99)

            scaling_factor = factors.get(tm_factor_name)
            link_cap_scaling_factors.append(scaling_factor)

        logging.debug("all possible scaling factors: {}".format(link_cap_scaling_factors))
        # Try each of the scaling factors in decreasing order until we find the
        # largest (acceptable) one that doesn't overload any link
        for scaling_factor in sorted(link_cap_scaling_factors, reverse=True):
            for link_util in utils_all:
                if link_util * scaling_factor > 1:
                    break
                    #This scaling factor would overload some link
            else:
                acceptable_scaling_factor = scaling_factor

        logging.debug("acceptable factor: {}".format(acceptable_scaling_factor))

        # This exception is thrown when the scaling factor is to aggressive
        # and overloads a link.
        assert acceptable_scaling_factor is not None, \
               "No acceptable scaling factor found"
        return acceptable_scaling_factor

    def __scale_traffic(self, tm_scaling_factor):
        """Multiply each node's endpoints' traffic volume by the traffic matrix
        scaling factor"""

        for port in self.ports():
            port.scale_traffic(tm_scaling_factor)

    def __find_end_points(self, how_many_end_points, radius):
        """ return a list of list of nodes that have sizes in sizelist """
        randomnode = random.choice(self.graph.nodes())

        endpoints = []
        termination_criterion, how_many_end_points = self.__find_end_points_recursive(randomnode, how_many_end_points, endpoints, radius)
        assert termination_criterion is True
        return endpoints

    def __find_end_points_recursive(self, node, how_many_end_points, endpoints, radius):
        """starting from node, search within a given radius for nodes that have
        endpoints. Terminate with True when we have found "how_many_end_points".
        Terminate with False when we reach the radius.

        Returns a list of the nodes found."""
        n = self.graph.node[node]
        if n['visited']:
            return (False, how_many_end_points)
        n['visited'] = True
        for i in range(n['rec_endpointcount']):
            if how_many_end_points > 0:
                assert len(n['endpoints']) > 0
                endpoints.append(n['endpoints'].pop())
                how_many_end_points -= 1
                n['rec_endpointcount'] -= 1
                if how_many_end_points <= 0:
                    return (True, how_many_end_points)

        radius -= 1
        if radius <= 0:
            return (False, how_many_end_points)

        for  i in nx.neighbors(self.graph, node):
            r, how_many_end_points = self.__find_end_points_recursive(i, how_many_end_points, endpoints, radius)
            if r is True:
                return (True, how_many_end_points)

        return (False, how_many_end_points)

    def routers(self):
        """Returns a list containing the routers in the network."""
        return [x for x, attr in self.graph.node.iteritems() if attr.get('is_router', False)]

    def ports(self):
        """Return a list of all the port instances"""
        return sorted(self.port.values())

    def __project_base_load(self):
        """ For every endpoint pair (endpoints are attributes in nodes, thus,
        between nodes), compute the shortest path between the endpoint's nodes,
        and add the traffic_vol attribute to the links along the path. """

        for src, dst, edgeattrs in self.graph.edges(data=True):
            if edgeattrs.get('load', 0) > 0:
                edgeattrs['load'] = 0
            if edgeattrs.get('load-solution', 0) > 0:
                edgeattrs['load-solution'] = 0
            edgeattrs['flows'] = {}

        # for every src switch, for every src end-point epsrc at src, for every
        # dst end-point epdst of epsrc project load on every link of the path
        # taken from src to the epdst's switch
        for src_port in self.ports():
            for dst_port_name, volume in src_port._volume_to.iteritems():
                dst_port = self.port[dst_port_name]
                src = src_port.switch
                dst = dst_port.switch
                path = self.paths[src][dst]
                links = zip(path[:-1], path[1:])
                for link in links:
                    e = self.graph.edge[link[0]][link[1]]
                    e.setdefault('load', 0)
                    e['load'] += volume
                    e.setdefault('load-solution', 0)
                    e['load-solution'] += volume
                    assert ((e['load'] * 8)/self.trace_duration) <= (e['link_capacity'] * 1000000), "overloaded link: "+str((link[0], link[1], e))

    def project_load_on_path(self, path, volume, util_thresh, cand=None, dst=None):
        """Atomically add volume to the edges of the graph over the path if
        doing so does not exceed the capacity "load-limit" of the link.

        util_thresh is a float in [0, 1]. It is the link capacity threshhold
        above which no link is allowed to be.

        Return True if success
        Return False if any link would be overloaded
        Traffic must never go negative
        """

        links = zip(path[:-1], path[1:])
        for link in links:
            edge_attrs = self.graph.edge[link[0]][link[1]]
            util = float(edge_attrs['load-solution'] + volume) / int(edge_attrs['load-limit'])

            # This traffic projection would cause this link to exceed the
            # threshhold, therefore, bail out.
            if util > util_thresh:
                return False

            assert int(edge_attrs['load-solution'] + volume) >= 0, "We made a link load-solution go negative. This should never happen." + """
            cand [{}] dst [{}] load [{}]
            path [{}]
            tpth [{}]
            link [{}]
            edge [{}]
            volume        [{}]
            load          [{}]
            load-limit    [{}]
            load-solution [{}]""".format(
                    cand,
                    dst,
                    cand.volume_to(dst),
                    path,
                    self.paths[path[0]][path[-1]],
                    link,
                    edge_attrs,
                    volume,
                    edge_attrs.get('load'),
                    edge_attrs.get('load-limit'),
                    edge_attrs.get('load-solution'))

        for link in links:
            edge_attrs = self.graph.edge[link[0]][link[1]]
            edge_attrs['load-solution'] += volume

        return True

    def draw(self):
        """Draw the topology"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib could not be found")
            return
        node_color = range(len(self.graph.nodes()))
        for i, node in enumerate(self.graph.nodes(data=True)):
            node, attr = node
            if attr.get('is_router', False):
                node_color[i] = 1000

        pos = nx.spring_layout(self.graph,iterations=200)
        nx.draw(self.graph,pos,node_color=node_color,
                node_size=[100*(nx.degree(self.graph,x)**1.25) for x in self.graph.nodes()],
                edge_color=['blue' for x,y,z in self.graph.edges(data=True)],
                edge_cmap=plt.cm.Blues,
                cmap=plt.cm.Blues)
        plt.show()

    def vitalstats(self):
        """Print vital statistics of the topology"""

        pd = EnterpriseData()
        print "{} configs in the Enterprise Dataset".format(len(pd.devices()))
        print "{} configs classified as L3 Devices".format(len(pd.l3devices()))
        print "{} configs classified as L2 Devices".format(len(pd.l2devices()))

        print "{} Nodes in the pruned graph".format(len(self.graph.nodes()))
        print "{} Nodes pruned from the graph".format(len(self.leaves))

        print "{} Edges in the network".format(sum([len(x) for x in self.graph.adjacency_list()]))
        print "{} Endpoints".format(self.numallendpoints)

    def shortest_path_via(self, src, dst, via=[], graph=None):
        """Return the shortest path from src to dst in self.graph via some other
        nodes. If no path exists, returns None"""

        if graph is None:
            graph = self.graph

        if type(via) is str:
            via = [via]

        try:
            if len(via) > 0:
                intermediate = via.pop(0)
                return nx.shortest_path(graph, src, intermediate)[:-1] + \
                    self.shortest_path_via(intermediate, dst, via, graph)
            else:
                return nx.shortest_path(graph, src, dst)
        except:
            return None

    def sorted_linkutil(self, data=False):
        """Returns sorted list of link utilizations"""
        trace_duration_secs = self.trace_duration
        bits_per_byte = 8
        bits_per_MBit = 1000000
        if data is True:
            return sorted([(((edge.get('load', 0) * bits_per_byte) / trace_duration_secs) /
                (edge['link_capacity'] * bits_per_MBit), src, dst, edge['link_capacity'])
                for src, dst, edge in self.graph.edges(data=True)])

        return sorted([((edge.get('load', 0) * bits_per_byte) / trace_duration_secs) /
            (edge['link_capacity'] * bits_per_MBit)
            for src, dst, edge in self.graph.edges(data=True)])
