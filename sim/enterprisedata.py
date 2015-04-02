#!/usr/bin/python
# Dan Levin

from collections import Counter, OrderedDict
from itertools import chain
import os
import re


class EnterpriseData(object):
    """
    Organizes the raw data from enterprise network datasets into python
    datastructures
    """
    def __init__(self, datadir="./topo-data/data-enterprise/configs/"):
        self.devicefiles = [datadir+"/"+i
                for i in os.listdir(datadir)
                if i.startswith("config")]

        self.__device = OrderedDict()
        for filename in sorted_nicely(self.devicefiles):
            devicename = os.path.basename(filename)
            self.__device[devicename] = Device(filename)

        assert len(self.devices()) == 1646
        # Total number ports that reference another in the description field
        assert sum([len(d.l2adjacencies()) for d in self.devices()]) == 4201
        # Total number ports that reference another in the
        # description field and have a positive-valued link capacity
        assert len(self.links()) == 3912
        # Total number of unique devices in l2 adjacencies
        assert len(set(chain(*[d.l2adjacencies() for d in self.devices()]))) == 1552
        # Total number of interfaces on all devices
        assert sum([len(d.ports) for d in self.devices()]) == 81091

    def devices(self):
        """Returns the devices in the network."""
        return self.__device.values()

    def l2devices(self):
        """Devices that do not exhibit eigrp or ospf configuration directives"""
        return [dev for dev in self.__device.values() if not (dev.has_eigrp or dev.has_ospf) ]

    def l3devices(self):
        """Devices that do exhibit eigrp and/or ospf configuration directives"""
        return [dev for dev in self.__device.values() if dev.has_eigrp or dev.has_ospf]

    def device(self, name):
        """Gets a device given a name"""
        return self.__device.get(name)

    def ports(self, name):
        """Returns the ports of the device name"""
        return map(str,self.__device.get(name).ports)

    def phys_ports(self, name):
        """return the names of ports for physical ports -- those which have
        "Ethernet" in their name and have a speed > 0"""
        return [port for port in self.__device.get(name).ports if port.speed > 0]


    def l3switches(self):
        """Return the l3switches in the network"""
        return [d for d in self.devices() if d.has_eigrp or d.has_ospf]

    def links(self):
        """Returns a dict of annotated links in the topology"""
        src2dst = OrderedDict()
        links = []
        [src2dst.setdefault(d, []).extend(d.links()) for d in self.devices()]

        for key, values in src2dst.iteritems():
            for value in values:
                links.append(
                    dict(
                        src=str(key),
                        dst=str(value.l2adjacency),
                        link_capacity=int(value.speed),
                        is_router=(key.has_eigrp or key.has_ospf),
                        portcount=int(key.number_of_ports()),
                        portname=value.name
                    )
                )

        return links

class Device(object):
    """Organizes metadata from each device level configuration file"""
    def __init__(self, filename):
        self.name = os.path.basename(filename)
        self.ports = []
        self.has_eigrp = False
        self.has_ospf = False
        self.readConfig(filename)

    def __str__(self):
        return self.name

    def readConfig(self, filename):
        """Parses a device config file"""
        configfile = open(filename, 'r')

        port = speed = dst = address = None
        for line in configfile.readlines():
            if line.startswith("interface"):
                tokens = line.split('interface ')
                portname = tokens[1]
                if port is not None:
                    port.speed = speed
                    port.l2adjacency = dst
                    port.address = address
                    self.ports.append(port)
                    speed = dst = address = None
                port = Port(self, portname.strip())
                if portname.startswith("TenGigabitEther"):
                    speed = 10000
                elif portname.startswith("GigabitEther"):
                    speed = 1000
                elif portname.startswith("FastEther"):
                    speed = 100
                elif portname.startswith("Ethernet"):
                    speed = 10
                elif portname.startswith("ethernet"):
                    speed = 10
                elif portname.startswith("LongReachEthernet"):
                    speed = 10
                else:
                    speed = 0
            elif line.startswith(" description"):
                tokens = line.split('description ')
                description = tokens[1]
                if description.startswith("config"):
                    dst = description.strip()
            elif line.startswith(" ip address"):
                tokens = line.split('address ')
                address = tokens[1].strip()
            elif line.startswith(" ip ospf"):
                self.has_ospf = True
            elif line.startswith("router eigrp"):
                self.has_eigrp = True

        # Add the last port
        if port is not None:
            port.speed = speed
            port.l2adjacency = dst
            port.address = address
            self.ports.append(port)

    def number_of_ports(self):
        """Returns the number of ports on the device"""
        return len(self.ports)

    def addresses(self):
        """Returns a list of addresses from this device"""
        return [p.address for p in self.ports if p.address is not None]

    def l2adjacencies(self):
        """Adjacent devices as specified by an interface with a description
        that references another device's name. These may include virtual
        interfaces such as VLANs and Port-channels"""
        return [p.l2adjacency for p in self.ports if p.l2adjacency is not None]

    def l2adjports(self):
        """Returns a list of ports that are adgenect to l2 switches"""
        return [p for p in self.ports if p.l2adjacency is not None]

    def links(self):
        """ l2 adjacencies which have a port speed > 0 """
        return [p for p in self.ports if p.l2adjacency is not None and p.speed > 0]


class Port(object):
    """switch or router port"""

    def __init__(self, device, name):
        self.device = device
        self.name = name.replace(" ","_")
        self.speed = None
        self.type = None
        self.l2adjacency = None
        self.l3adjacency = None
        self.address = None

    def __str__(self):
        return "{n} {s} {c} {a}".format(n=self.name,
                s=self.speed, c=self.l2adjacency, a=self.address)

class DirectedLink(object):
    """Link between two Ports"""

    def __init__(self, port1, port2):
        self.from_port, self.to_port = (port1, port2)

    def __str__(self):
        return "{d1} {p1} {s1} {d2} {p2} {s2}".format(
                p1=str(self.from_port), p2=str(self.to_port),
                d1=str(self.from_port.device), d2=str(self.to_port.device),
                s1=str(self.from_port.speed), ds=str(self.to_port.speed))


def testAdjacencySymmetry(data):
    """
    Determine which l2adjacencies are symmetric and if not, by how much
    For example for ports a, b, show:
    a->b but not b->a
    a->b a->b (twice) but b->a (once)
    """
    srcdst = [(a['src'], a['dst']) for a in data.links()]
    expected = Counter([(b,a) for a,b in srcdst])
    existing = Counter(srcdst)
    symmetric = [(sd, existing[sd], expected[sd]) for sd in srcdst if existing[sd] == expected[sd]]
    weirdos = [(sd, existing[sd], expected[sd]) for sd in srcdst if existing[sd] != expected[sd]]

    assert len(symmetric) + len(weirdos) == len(data.links())
    return (symmetric, weirdos)

def sorted_nicely(l):
    """Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

