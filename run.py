#!/usr/bin/env python3
# encoding: utf-8

# File        : run.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Feb 27
#
# Description : 

import os
import uproot
import argparse
import numpy as np
import awkward as ak
import pprint
import glob
import subprocess
from hist import Hist
# from EMTFHits import EMTFHits
from EMTFTrack import EMTFTracks
from OMTFTrack import OMTFTracks
# from Hybrid import HybridStub
from TrackerMuons import TrackerMuons
from SAMuons import SAMuons
# from L1Tracks import L1Tracks
# from GMTStub import GMTStubs

import mplhep as hep
hep.style.use("CMS")
# hep.cms.label("Phase 2", data=False, loc=0)
eosfolder = '/eos/uscms/store/user/lpctrig/benwu/GMT_Nano/Spring23_GMTv8_v0/'
samplemap = {
    # "DsToTauTo3Mu":  "DsToTauTo3Mu_TuneCP5_14TeV-pythia8",
    "DYToLL"      :  "DYToLL_M-50_TuneCP5_14TeV-pythia8",
    "MinBias"     :  "MinBias_TuneCP5_14TeV-pythia8",
    "H24Mu"       :  'HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8',
    # "Muon200"     :  "SingleMuon_Pt-0To200_Eta-1p4To3p1-gun",
    # "Muon500"     :  "SingleMuon_Pt-200To500_Eta-1p4To3p1-gun",
    # "TauTo3Mu"    :  "TauTo3Mu_TuneCP5_14TeV-pythia8",
    # "TTTo2L2Nu"   :  "TTTo2L2Nu_TuneCP5_14TeV-powheg-pythia8",
    # "TTToSemi"    :  "TTToSemiLepton_TuneCP5_14TeV-powheg-pythia8"
}

def eosls(sample):
    hostname =os.uname().nodename
    if sample not in samplemap.keys():
        return None
    if "local" in hostname:
        outputLocation = "/Users/benwu/Work/L1MuNanoAna/V9Data/"
    else:
        outputLocation = eosfolder + samplemap[sample]
    if "eos" in outputLocation:
        p = subprocess.Popen("eos root://cmseos.fnal.gov find %s -type f -name \'*.root\' " % outputLocation,
                                      stdout=subprocess.PIPE, shell=True)
        (dummyFiles_, _) = p.communicate()
        dummyFiles = str(dummyFiles_, 'UTF-8').split("\n")
        dummyFiles = [ f for f in dummyFiles if f.endswith(".root") ]
    else:
        # dummyFiles = [ outputLocation +"/"+i for i in os.listdir(outputLocation)]
        dummyFiles = glob.glob(outputLocation +"/%s*.root" % sample)
    return dummyFiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Running over EMTF Ntuple')
    parser.add_argument("-s", "--sample", dest="sample", default='test',
                        help="sample")
    args = parser.parse_args()

    if args.sample == "test":
        # filelist = ["./H24mu_v8.root"]
        filelist = ["./V9Data/H24Mu.root"]
        # filelist = ["./DYLL_v8.root"]
        # filelist = ["./DYLL_GMTV7.root"]
        # filelist = ["./DYLL_GMTV5.root"]
        # filelist = ["./l1nano_v5.root"]
        # filelist = ["./l1nano.root"]
    else:
        filelist = eosls(args.sample)
    filename = [i+":Events" for i in filelist]
    events = uproot.iterate(
        filename,
        # step_size is still important
        # step_size="20 MB",
        step_size="2 GB",
        # options you would normally pass to uproot.open
        # xrootd_handler=uproot.MultithreadedXRootDSource,
        # num_workers=10,
    )

    ### Store the output file
    outfile = uproot.recreate("%s_hists.root" % args.sample )

    # mod_tkmuons = TrackerMuons()
    mod_samuons = SAMuons("SA", "samu_", isDisplaced=True, matchdR=0.3)
    mod_sadisp3 = SAMuons("SADisp_dR3", "dismu_", isDisplaced=True, matchdR=0.3)
    # mod_sadisp6 = SAMuons("SADisp_dR6", "dismu_", isDisplaced=True, matchdR=0.6)
    # mod_fwds = SAMuons("Fwd", "fwdmu_" , isDisplaced=True, matchdR=0.3)
    # mod_fwdisp3 = SAMuons("FwdDisp_dR3", "fwddismu_", isDisplaced=True, matchdR=0.3)
    # mod_fwdisp6 = SAMuons("FwdDisp_dR6", "fwddismu_", isDisplaced=True, matchdR=0.6)
    # mod_emtf = EMTFTracks()
    # mod_omtf = OMTFTracks()
    modules = [
        # mod_hit,
        # mod_emtf,
        # mod_hbstub,
        mod_samuons,
        mod_sadisp3,
        # mod_sadisp6,
        # mod_fwds,
        # mod_fwdisp3,
        # mod_fwdisp6,
        # mod_emtf, 
        # mod_omtf,
        # mod_tkmuons,
        # mod_l1trks,
    ]

    # ### Running over samples
    nTotal = 0
    for e in events:
        nEvent = len(e)
        print("Processing %d events" % nEvent)
        nTotal += nEvent
        [m.run(e) for m in modules]
        # ## Running on local Hybrid Stubs
        # # hb = mod_hbstub.GetHb()
        # # # mod_l1trks.runlocalhb(hb)
        # # mod_emtf.runEMTFHybrid(hb)

    ### End of run
    # if "MinBias" not in args.sample :
        # nTotal = 0
    [m.endrun(outfile, nTotal) for m in modules]
