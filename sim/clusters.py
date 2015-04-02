#!/usr/bin/env python

import argparse
from random import seed

LBNL_PREFIXES = [
    u'0.0.0.0/255.255.255.255',
    u'10.0.0.0/255.0.0.0',
    u'128.3.0.0/255.255.0.0',
    u'128.55.0.0/255.255.0.0',
    u'131.243.0.0/255.255.0.0',
    u'169.254.0.0/255.255.0.0',
    u'172.16.0.0/255.240.0.0',
    u'192.168.0.0/255.255.0.0',
    u'198.125.133.0/255.255.255.0',
    u'198.128.24.0/255.255.252.0',
    u'198.129.88.0/255.255.252.0',
    u'255.255.255.255/255.255.255.255']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                        help="initialize the random seed",
                        default=1,
                        type=int)
    parser.add_argument("--exclude", "-e",
                        help="exclude switches in the topo from upgrade in the heuristic",
                        nargs='+',
                        default=[],
                        type=int,
                        dest="exclude")
    parser.add_argument("--trace",
                        help="trace json file to read graph input",
                        default="tm2004.json",
                        type=str)

    args = parser.parse_args()

    seed(args.seed)

def getcandidate(k):
    """This method takes the first to octals of an IP as candidate cluster
    name."""
    candidate = ".".join(k.split(".")[:2])
    if candidate.startswith(u'172.'):
        candidate = u'172.16'
    if candidate.startswith(u'10.'):
        candidate = u'10.0'
    return candidate

def get_clusters(lbnl):
    """This method clusters the IPs in the tm lbnl. Clusters with less than 66
    hosts are aggregated in the 'aggregate' cluster"""
    clusters = {}
    clusters[u'gateway'] = 0
    clusters[u'aggregate'] = 0

    for prfx in LBNL_PREFIXES:
        clusters[".".join(prfx.split(".")[:2])] = 0
    for k in lbnl.iterkeys():
        candidate = getcandidate(k)
        clusters[candidate] += 1

    del(clusters[u'0.0'])
    del(clusters[u'255.255'])
    del(clusters[u'172.16'])

    for k,v in clusters.iteritems():
        if v < 66 and k != 'aggregate' and k != 'gateway':
            clusters['aggregate'] += v
    newclusters = {}
    for k,v in clusters.iteritems():
        if v > 66 or k == 'aggregate' or k == 'gateway':
            newclusters[k] = v

    clusters = newclusters

    clusters_to_ips = {}
    for k in lbnl.iterkeys():
        candidate = getcandidate(k)
        if candidate.startswith(u'0.0'):
            continue
        if candidate not in clusters:
            candidate = 'aggregate'

        # a map of all the ip addresses in a cluster
        clusters_to_ips.setdefault(candidate, []).append(k)

    return clusters_to_ips

if __name__ == "__main__":
    main()
