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
from Common import *

'''
 'samu_charge',
 'samu_chargeNoPh',
 'samu_d0',
 'samu_eta',
 'samu_hwBeta',
 'samu_hwEta',
 'samu_hwPhi',
 'samu_hwPt',
 'samu_hwQual',
 'samu_phi',
 'samu_pt',
 'samu_z0',
'''


class SAMuons(Module):
    def __init__(self, name="SAMuons", prefix="samu_", isDisplaced = False,
                 matchdR=0.3):
        super().__init__(name, isDisplaced, matchdR)
        self.prefix=prefix
        self.__bookRate()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith(self.prefix):
                setattr(self, k.split(self.prefix)[-1], event[k])
        if not hasattr(self, 'pt'):
            setattr(self, 'pt', self.hwPt * GMT_LSB_Pt)
        if not hasattr(self, 'eta'):
            setattr(self, 'eta', self.hwEta * GMT_LSB_cor)
        if not hasattr(self, 'phi'):
            setattr(self, 'phi', self.hwPhi * GMT_LSB_cor)
        super().__GetEvent__(event)

    def __bookRate(self):
        self.__bookobj()
        self.BookEff()

    def __bookobj(self, name=None):
        objhist = {
            "pt" :  Hist.new.Reg(100, 0, 100, name="pt").Double(),
            "phi" :  Hist.new.Reg(80, -4, 4, name="phi").Double(),
            "eta" :  Hist.new.Reg(90, -3, 3, name="eta").Double(),
            "rate" : Hist.new.Reg(100, 0, 100, name="rate").Double(),
            "qual" : Hist.new.Reg(16, 0, 16, name="qual").Double(),
            "Endcap_phi" :  Hist.new.Reg(80, -4, 4, name="phi").Double(),
            "Endcap_phi2" :  Hist.new.Reg(80, -4, 4, name="phi").Double(),
        }
        self.h.update(objhist)

        ratehists = {}
        for i in range(12):
            ratehists["rate_qual%d" % i] = Hist.new.Reg(100, 0, 100, name="rate_qual%d" % i).Double()
            for region, etas in MuonEtamap.items():
                ratehists["%s_rate_qual%d" % (region, i)] = Hist.new.Reg(100, 0, 100, name="%s qual>% d"% (region , i)).Double()
        self.h.update(ratehists)

    def __fillRate(self):
        self.h["pt"].fill_flattened(self.pt)
        self.h["phi"].fill_flattened(self.phi)
        self.h["eta"].fill_flattened(self.eta)
        self.h["rate"].fill_flattened(self.pt)
        self.h["Endcap_phi"].fill_flattened(self.phi[abs(self.eta)>1.2])
        self.h["Endcap_phi2"].fill_flattened(self.hwPhi[abs(self.eta)>1.2] * GMT_LSB_cor)
        for i in range(12):
            self.h["rate_qual%d" % i].fill_flattened(self.pt[self.hwQual >= i])
            for region, etas in MuonEtamap.items():
                sel = (self.hwQual >= i) & (abs(self.eta)>= etas[0]) & (abs(self.eta) < etas[1])
                self.h["%s_rate_qual%d" % (region, i)].fill_flattened(self.pt[sel])


    def BookEff(self):
        self.h["effemu"] = Hist.new.Reg(100, 0, 100, name="rate").Double()
        self.h["emu"] = Hist.new.Reg(100, 0, 100, name="rate").Double()
        self.h["eff_eta_de"] =  Hist.new.Reg(90, -3, 3, name="eta;eta;y").Double()
        self.h["eff_eta_nu"] =  Hist.new.Reg(90, -3, 3, name="eta;eta;y").Double()

    def CalEff(self):
        super().__CalDefaultEff__()

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()
        self.CalEff()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
