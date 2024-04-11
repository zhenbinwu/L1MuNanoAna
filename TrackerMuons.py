#!/usr/bin/env python
# encoding: utf-8

# File        : hybrid.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Mar 23
#
# Description : 

import uproot
import numpy as np
import awkward as ak
from hist import Hist
from Common import *

""" Variables in EMTFNtuple
'ntkmu',
'tkmu_hwBeta',
'tkmu_hwEta',
'tkmu_hwPhi',
'tkmu_hwPt',
'tkmu_hwQual',
'tkmu_charge',
'tkmu_chargeNoPh',
'tkmu_hwIso',
'tkmu_d0',
'tkmu_eta',
'tkmu_phi',
'tkmu_pt',
'tkmu_z0',
""" 


class TrackerMuons(Module):
    def __init__(self, name="TkMuons"):
        super().__init__(name)
        self.__bookRate()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith("tkmu_"):
                setattr(self, k.split("tkmu_")[-1], event[k])
        super().__GetEvent__(event)

    def __bookRate(self):
        self.h.update({
            "tkmu_rate" : Hist.new.Reg(100, 0, 100, name="rate").Double(),
            "tkmu_qual" : Hist.new.Reg(1024, 0, 1024, name="qual").Double(),
            "tkmu_isosum" : Hist.new.Reg(1024, 0, 1024, name="isosum").Double(),
        })

    def CalEff(self):
        super().__CalDefaultEff__()

    def __fillRate(self):
        self.h["tkmu_rate"].fill(ak.flatten(self.pt))
        self.h["tkmu_qual"].fill(ak.flatten(self.hwQual))
        self.h["tkmu_isosum"].fill(ak.flatten(self.hwIso))

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()
        self.CalEff()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
