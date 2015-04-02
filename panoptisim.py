#!/usr/bin/env python2.7
#
# Dan Levin <dan@net.t-labs.tu-berlin.de>

import argparse
import gzip
import logging
import logging.config
import os
import pickle
import random
import subprocess
import time

from os import getpid
from socket import gethostname

from sim.simulator import Simulator
from sim.topology import EnterpriseTopology


def get_parser():
    """Returns the parser for Panoptisim"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_1 = subparsers.add_parser('new',
                                      help="start a new simulation run using \
                                        cmdline specified paramters")
    parser_2 = subparsers.add_parser('resume',
                                      help="resume a partially-run simulation \
                                      using paramters obtained from a json \
                                      results file")
################################################################################
# Following paramters must be determined by json when using file-recovery
################################################################################
    for myparser in [parser_1, parser_2]:
        # Only require the following args if we're running "new"
        isrequired = (myparser == parser_1)

        myparser.add_argument("--seedmapping",
                            help="random seed for endpoint mapping",
                            required=isrequired,
                            type=int)
        myparser.add_argument("--seednextswitch",
                            help="random seed for RAND nextswitch",
                            required=isrequired,
                            type=int)
        myparser.add_argument("--tmsf",
                            help="TM scaling factor",
                            #choices=['1.0', 'max-50', '1-50', '0.5-50'],
                            # Ony 1.0 and max-50 are sensible options, see
                            # EnterpriseTopology.__calculate_traffic_scaling_factor
                            choices=['1.0', 'max-50'],
                            required=isrequired,
                            type=str)
        myparser.add_argument("--epsf",
                            help="end-point scaling factor",
                            required=isrequired,
                            type=int)
        myparser.add_argument("--epp",
                            help="entries-per-port: number of 'policy' flow-table \
                            entries to consume per communicating src-dst pair",
                            required=isrequired,
                            type=int)
        myparser.add_argument("--tm",
                            help="lbnl traffic matrix year",
                            required=isrequired,
                            choices=['2004', '2005'],
                            type=str)
        myparser.add_argument("--pattern",
                            # Now deprecated and not used anymore
                            help="which communication pattern to consider, \
                            deprecated and not used anymore",
                            #required=isrequired,
                            choices=['all-to-all', 'only-tm'],
                            default='only-tm',
                            type=str)
        myparser.add_argument("--switchstrategy",
                            help="heuristic for which switch to consider next",
                            required=isrequired,
                            choices=['VOL', 'DEG', 'RAND', 'ROUTERS'],
                            type=str)
        myparser.add_argument("--portstrategy",
                            help="in what order to try to accommodate sdn ports",
                            required=isrequired,
                            choices=['default', 'sdn-switches-first'],
                            type=str)
        myparser.add_argument("--maxvlans",
                            help="max vlans limit",
                            required=isrequired,
                            type=int)
        myparser.add_argument("--maxft",
                            help="max flow table limit",
                            required=isrequired,
                            type=int)
        myparser.add_argument("--activefrontiersize",
                            help="limit the max size of the active frontier",
                            default=None,
                            type=int)

################################################################################
        isrequired = (myparser == parser_2)
        myparser.add_argument("--resumefromfile",
                            help="resume a simulation run using the paramters \
                                  obtained from a json results file",
                            required=isrequired,
                            type=str)
################################################################################
# Always Required
################################################################################
        myparser.add_argument("--toupgrade",
                            help="how many switches to upgrade",
                            required=True,
                            type=int)
        myparser.add_argument("--numprioports",
                            help="number of ports that must be SDN",
                            default=0,
                            type=int)
        myparser.add_argument("--exclude", "-e",
                            help="exclude switches from upgrade",
                            nargs='+',
                            default=['config1515', 'config775', 'config842'],
                            type=str)
        myparser.add_argument("--upgraded",
                            help="consider the switches in this list as \
                            already upgraded",
                            nargs='+',
                            default=[],
                            type=str)
        myparser.add_argument("--onlyupgrade",
                            help="only upgrade the following switches in the \
                            given order",
                            nargs='+',
                            default=[],
                            type=str)
        myparser.add_argument("--linkutilthresh",
                            help="max link util before rejecting an SDN port's \
                            traffic",
                            default=1,
                            type=float)
        myparser.add_argument("--magicswitches",
                            help="just upgrade n of the magic switches in \
                            random (seednextswitch) order",
                            default=0,
                            type=int)



################################################################################
# Following paramters should be taken from json during file recovery, but may be
# overridden from json when using file-recovery.
################################################################################
    parser.add_argument("--stop",
                        help="stop after all ports are made SDN",
                        action='store_true',
                        default=False)
    parser.add_argument("--outputdir",
                        help="Directory where json output is written",
                        type=str,
                        default=os.getcwd())
    parser.add_argument("--gitid",
                        help="git commit id of the running experiment",
                        default=subprocess.check_output(
                            ["git log | head -1"],shell=True
                        ).strip().replace("commit ",""),
                        type=str)
    parser.add_argument("--starttime",
                        help="the time at which the simulation begins execution",
                        default=str(time.time()),
                        type=str)
    parser.add_argument("--hostname",
                        help="the hostname on which the simulation is run: \
                            This only documents the hostname in the filename.",
                        default=gethostname(),
                        type=str)
    parser.add_argument("--pid",
                        help="the simulatoin process id",
                        default=str(getpid()),
                       type=str)
################################################################################
# Following paramters are optional and may be specified regardless of subcommand
################################################################################
    parser.add_argument("--pickletopo",
                        help="use a pickeled topo to speed up reading topology",
                        default=False,
                        action='store_true')
    parser.add_argument("--verbosity", "-v",
                        help="set the logging level",
                        choices=['DEBUG', 'INFO', 'WARN', 'ERROR'],
                        default='INFO',
                        type=str)

    return parser


def main():
    FORMAT="%(asctime)s %(process)d %(levelname)s %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('panoptisim')

    parser = get_parser()
    args = parser.parse_args()

    logger.setLevel(args.verbosity)

    for module in ['sim.topology', 'sim.simulator']:
        lg = logging.getLogger(module)
        lg.setLevel(logger.getEffectiveLevel())

    logger.info("Starting on host [{}]".format(args.hostname))
    logger.info("Starting with args: {args}".format(
        args="\n\t".join(sorted([str(k)+": "+str(v) for k,v in
            args.__dict__.iteritems()]))
        )
    )

    logger.debug("Excluding Devices: %s" % str(args.exclude))
    random.seed(args.seedmapping)
    logger.debug("Mapping Randomseed is: %s" % str(args.seedmapping))
    EPscalingFactor = int(args.epsf)

    topo = None
    if args.pickletopo:
        topo = EnterpriseTopology.load_from_pickle(args.tm, args.tmsf,
                                                   args.seedmapping,
                                                   EPscalingFactor)
    else:
        logger.info("Creating New EnterpriseTopology")
        topo = EnterpriseTopology(tm=args.tm, tm_factor_name=args.tmsf,
                              EPscalingFactor=EPscalingFactor)

    random.seed(args.seednextswitch)
    logger.debug("Nextswitch Randomseed: %s" % str(args.seednextswitch))

# Magic switches are switches that can accommmodate all 1296 SDN ports with
# traffic under epp=2 maxvlan=1024 maxft=100000
    if args.magicswitches > 0:
        magic_switches = ['config451', 'config1520', 'config333', 'config424',
        'config426', 'config427', 'config428', 'config429', 'config431',
        'config433', 'config434', 'config435', 'config436', 'config437',
        'config439', 'config441', 'config442', 'config444', 'config445',
        'config446', 'config447', 'config449', 'config450', 'config452',
        'config453', 'config457', 'config458', 'config459', 'config460',
        'config461', 'config462', 'config463', 'config465', 'config468',
        'config469', 'config471', 'config474', 'config567', 'config714',
        'config775', 'config823', 'config842', 'config962', 'config1515',
        'config455', 'config473', 'config470', 'config418', 'config467',
        'config472', 'config713']
        random.shuffle(magic_switches)
        args.onlyupgrade = magic_switches[:args.magicswitches]
        logger.info("Starting with magic switches: {args}".format(
            args="\n\t".join(sorted([str(k)+": "+str(v) for k,v in
                args.__dict__.iteritems()]))
            )
        )


    simulation = Simulator(topo, args)
    simulation.run()

    logger.info("Finished Simulation")

if __name__ == '__main__':
    main()
