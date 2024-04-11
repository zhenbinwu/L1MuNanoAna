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

from Util import *
'''
 'emtf_Q',
 'emtf_beta',
 'emtf_d0',
 'emtf_endcap',
 'emtf_eta',
 'emtf_phi',
 'emtf_pt',
 'emtf_qual',
 'emtf_sector',
 'emtf_z0',

'''

class EMTFTracks(Module):
    def __init__(self, name="EMTF", prefix="emtf_"):
        super().__init__(name)
        self.prefix=prefix
        self.__bookRate()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith(self.prefix):
                setattr(self, k.split(self.prefix)[-1], event[k])
        ## Follow the code from FwdMuonTranslator
        self.pt = self.pt * GMT_LSB_Pt
        ## Understand the phi conversion
        locrad = calc_phi_loc_rad_from_int(self.phi)
        self.phi = calc_phi_glob_rad_from_loc(locrad, self.sector)
        theta = calc_theta_rad_from_int(self.eta)
        self.eta =self.endcap*calc_eta_from_theta_rad(theta)
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
        }
        self.h.update(objhist)

    def __fillRate(self):
        self.h["pt"].fill_flattened(self.pt)
        self.h["phi"].fill_flattened(self.phi)
        self.h["eta"].fill_flattened(self.eta)
        self.h["rate"].fill_flattened(self.pt)


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
