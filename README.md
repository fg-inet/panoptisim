Panoptisim
======================

Panoptisim is a tool for planning a partial SDN deployment for
[Panopticon](http://perso.uclouvain.be/marco.canini/papers/panopticon.TR.pdf).
Given a topology, packet-level traffic traces, and various resource
constraints, Panoptisim computes the locations to deploy SDN switches in the
given network topology.

# Installation

To install Panoptisim, Python 2.7 needs to be installed on the system. Then it
can be installed by executing

```
python2 setup.py install
```

which will pull the necessary dependencies and install Panoptisim.


# Usage
To get an overview about the supported parameters of Panoptisim, run
```
panoptisim.py --help
```
To use the included sample topology dataset, you need to specify
```
panoptisim.py --pickletopo
```
To start a new simulation, the keyword _new_ needs to be supplied to Panoptisim.
To get an overview of the parameters for running a new simulation, run
```
panoptisim.py new --help
```

The following example command should run out of the box:
```
./panoptisim.py --pickle new --seedmapping 1 --seednextswitch 1 --tmsf max-50 --epsf 1 --epp 10 --tm 2004 --switchstrategy RAND --portstrategy default --maxvlans 512 --maxft 100000 --toupgrade 100
```
It will begin a simulation based upon the supplied picked topology annotated
based on the supplied parameters (seedmapping 1, seednextswitch 1, tmsf max-50,
and tm 2004). It will iterate over 100 switches in random fashion, at each
step, trying to satisfy SDNc ports subject to vlan flow table and traffic
constraints.

# Structure
Panoptisim consists of three main classes. A input parser for the command line,
a class for holding an internal representation of a resource-annotated network
topology and a simulator that determines the hybrid deployment. Additionally,
there is support for parsing device-level configuration files (cited in the
publication) to build the topology.

## panoptisim.py
This file contains the argparse parser, documentation on each argparse option.
It is the starting point for parsing the topology and the entering the
simulator main loop.

## EnterpriseTopology
This class implements network topology including an
annotated NetworkX graph instance that represents directed link connectivity,
link capacity, and switch metadata. The topology keeps state of the network
link utilization as the simulation runs.

## EnterpriseData
Responsible for parsing device level configuration files into Python data
structures.  Used by EnterpriseTopology to build the topology. Due to license
issues, we are not able to provide the original device-level configuration
data. Instead, we provide pickled EnterpriseTopology instances which we
generated using this EnterpriseData class.

## Simulator
This is the class responsible for running the simulation.


======================
# Authors

* Dan Levin dan@badpacket.in
* Fabian Schaffert fabian@badpacket.in
* Marco Canini marco@badpacket.in
