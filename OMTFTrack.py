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
 'omtf_Q',
 'omtf_bx',
 'omtf_hwDXY',
 'omtf_hwEta',
 'omtf_hwPhi',
 'omtf_hwPt',
 'omtf_hwPtUnc',
 'omtf_hwQual',
 'omtf_muIdx',
 'omtf_processor',
'''

class OMTFTracks(Module):
    def __init__(self, name="OMTF", prefix="omtf_"):
        super().__init__(name)
        self.prefix=prefix
        self.__bookRate()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith(self.prefix):
                setattr(self, k.split(self.prefix)[-1], event[k])
        self.pt = self.hwPt * 0.5  #Phase-1 LSB
        self.eta = self.hwEta * 0.010875 #Phase-1 LSB
        globphi = self.__uGMT_calGlobalPhi()
        self.phi = globphi * (2*math.pi/576) # Phase-1 LSB
        super().__GetEvent__(event)

    def __uGMT_calGlobalPhi(self):
        globphi = self.processor *192+ self.hwPhi
        globphi = (globphi + 600) % 576
        return globphi


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
            "processor" : Hist.new.Reg(16, 0, 16, name="processor").Double(),
            "muIdx" : Hist.new.Reg(16, 0, 16, name="muIdx").Double(),
            "hwpt" :  Hist.new.Reg(250, 0, 500, name="pt").Double(),
            "hwphi" :  Hist.new.Reg(300, -150, 150, name="phi").Double(),
            "hweta" :  Hist.new.Reg(300, -150, 150, name="eta").Double(),
            "globphi" :  Hist.new.Reg(120, -600, 600, name="phi").Double(),
        }
        self.h.update(objhist)

    def __fillRate(self):
        self.h["pt"].fill(ak.flatten(self.pt))
        self.h["phi"].fill(ak.flatten(self.phi))
        self.h["eta"].fill(ak.flatten(self.eta))
        self.h["rate"].fill(ak.flatten(self.pt))
        self.h["hwpt"].fill(ak.flatten(self.hwPt))
        self.h["hwphi"].fill(ak.flatten(self.hwPhi))
        self.h["hweta"].fill(ak.flatten(self.hwEta))
        self.h["processor"].fill(ak.flatten(self.processor))
        self.h["muIdx"].fill(ak.flatten(self.muIdx))


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
