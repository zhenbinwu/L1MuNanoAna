#!/usr/bin/env python3.11
# encoding: utf-8

# File        : run.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Feb 27
#
# Description :

import argparse
import glob
import os
import pprint
import subprocess
from typing import DefaultDict

import awkward as ak
import mplhep as hep
import numpy as np
import uproot
from hist import Hist

# from EMTFHits import EMTFHits
from EMTFTrack import EMTFTracks
from OMTFTrack import OMTFTracks
from SAMuons import SAMuons

# from L1Tracks import L1Tracks
from Tau23Mus import TauModules

# from Hybrid import HybridStub
from TrackerMuons import TrackerMuons

# from GMTStub import GMTStubs

hep.style.use("CMS")
# hep.cms.label("Phase 2", data=False, loc=0)

eosfolder = "/eos/uscms/store/user/lpctrig/benwu/GMT_Nano/Spring23_GMTv8_v0/"
outputlocation = "./Plots/Hists/"
samplemap = {
    # "DsToTauTo3Mu":  "DsToTauTo3Mu_TuneCP5_14TeV-pythia8",
    "DYToLL": "DYToLL_M-50_TuneCP5_14TeV-pythia8",
    "MinBias": "MinBias_TuneCP5_14TeV-pythia8",
    "H24Mu900": "HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8",
    "H24Mu1k": "HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8",
    "H24Mu": "HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8",
    # "Muon200"     :  "SingleMuon_Pt-0To200_Eta-1p4To3p1-gun",
    # "Muon500"     :  "SingleMuon_Pt-200To500_Eta-1p4To3p1-gun",
    # "TauTo3Mu"    :  "TauTo3Mu_TuneCP5_14TeV-pythia8",
    # "TTTo2L2Nu"   :  "TTTo2L2Nu_TuneCP5_14TeV-powheg-pythia8",
    # "TTToSemi"    :  "TTToSemiLepton_TuneCP5_14TeV-powheg-pythia8"
    "H24Mu_v9_14DT": "./V9Data/HTo2LongLivedTo4mu_MFF_Fall22_ALL.root",
    "H24Mu_v9_13DT": "./V10Data/HTo2LongLivedTo4mu_Fall22_v10_temp.root",
    "H24Mu_v9_12DT": "./Data_IBv9_12DT/HTo2LongLivedTo4mu_Fall22_IBv9_12DT_temp.root",
    "DispMu10_12DT": "Data_IBv9_12DT/DispMu10_Fall22_IBv9_12DT.root",
    "DispMu2_12DT": "Data_IBv9_12DT/DispMu2_Fall22_IBv9_12DT.root",
    "DispMu30_12DT": "Data_IBv9_12DT/DispMu30_Fall22_IBv9_12DT.root",
    "H24Mu_12DT": "Data_IBv9_12DT/HTo2LongLivedTo4mu_Fall22_IBv9_12DT.root",
}


def eosls(sample):
    hostname = os.uname().nodename
    dummyFiles = None
    if sample not in samplemap.keys():
        return None
    ## Get the file folder
    if "local" in hostname:
        inputLocation = "/Users/benwu/Work/L1MuNanoAna/V9Data/"
    else:
        inputLocation = eosfolder + samplemap[sample]

    ## Get the files
    ## From EOS
    if "eos" in inputLocation:
        p = subprocess.Popen(
            "eos root://cmseos.fnal.gov find %s -type f -name '*.root' "
            % inputLocation,
            stdout=subprocess.PIPE,
            shell=True,
        )
        (dummyFiles_, _) = p.communicate()
        dummyFiles = str(dummyFiles_, "UTF-8").split("\n")
        dummyFiles = [f for f in dummyFiles if f.endswith(".root")]
        return dummyFiles
    ## From local
    ## If root file path is known
    if ".root" in samplemap[sample]:
        dummyFiles = [samplemap[sample]]
    if not dummyFiles:
        dummyFiles = glob.glob(inputLocation + "/%s*.root" % sample)
    if not dummyFiles:
        dummyFiles = [inputLocation + "/" + i for i in os.listdir(inputLocation)]
    print(dummyFiles)
    return dummyFiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running over EMTF Ntuple")
    parser.add_argument("-s", "--sample", dest="sample", default="test", help="sample")
    parser.add_argument(
        "-o", "--outfolder", dest="output", default="./", help="histogram output folder"
    )
    args = parser.parse_args()

    if args.sample == "test":
        # filelist = ["./H24mu_v8.root"]
        # filelist = ["./V9Data/H24Mu1k.root"]
        # filelist = ["./DYLL_v8.root"]
        # filelist = ["./DYLL_GMTV7.root"]
        # filelist = ["./DYLL_GMTV5.root"]
        # filelist = ["./test.root"]
        # filelist = ["./V9Data/HTo2LongLivedTo4mu_MFF_Fall22_ALL.root"]
        filelist = ["./TauData/tau_v0.root"]
        # filelist = ["./Data_IBv9_12DT/HTo2LongLivedTo4mu_Fall22_IBv9_12DT_temp.root"]
        # filelist = ["./V10Data/DYToLL_IBv9_13DT.root"]
    else:
        filelist = eosls(args.sample)
    filename = [i + ":Events" for i in filelist]
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
    outfile = uproot.recreate("%s/%s_hists.root" % (outputlocation, args.sample))

    mod_tkmuons = TrackerMuons(prefix="L1gmtTkMuon_")
    mod_tau = TauModules()
    # mod_samuons = SAMuons("SA", "L1GTgmtMuon_", isDisplaced=False, matchdR=0.3)
    # mod_samuons = SAMuons("SA", "L1GTgmtMuon_", isDisplaced=True, matchdR=0.3)
    # mod_samuons = SAMuons("SA", "L1MuonKMTF_", isDisplaced=True, matchdR=0.3)
    # mod_sadisp3 = SAMuons("SADisp_dR3", "L1DispMuonKMTF_", isDisplaced=True, matchdR=0.3)
    # mod_sadisp3 = SAMuons("SADisp_dR3", "L1GTgmtDispMuon_", isDisplaced=True, matchdR=0.3)
    # mod_sadisp6 = SAMuons("SADisp_dR6", "L1GTgmtDispMuon_", isDisplaced=True, matchdR=0.6)
    # mod_sadisp4 = SAMuons("SADisp_dR4", "L1GTgmtDispMuon_", isDisplaced=True, matchdR=0.4)
    # mod_samuons = SAMuons("SA", "samu_", isDisplaced=True, matchdR=0.3)
    # mod_sadisp3 = SAMuons("SADisp_dR3", "dismu_", isDisplaced=True, matchdR=0.3)
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
        # mod_samuons,
        # mod_sadisp3,
        # mod_sadisp6,
        # mod_fwds,
        # mod_fwdisp3,
        # mod_fwdisp6,
        # mod_emtf,
        # mod_omtf,
        # mod_tkmuons,
        mod_tau,
        # mod_l1trks,
    ]

    # ### Running over samples
    nTotal = 0
    for e in events:
        nEvent = len(e)
        print("Processing %d events" % nEvent)
        nTotal += nEvent
        [m.run(e) for m in modules]
        ## Running on local Hybrid Stubs
        # hb = mod_hbstub.GetHb()
        # # mod_l1trks.runlocalhb(hb)
        # mod_emtf.runEMTFHybrid(hb)

    ### End of run
    [m.endrun(outfile, nTotal) for m in modules]
