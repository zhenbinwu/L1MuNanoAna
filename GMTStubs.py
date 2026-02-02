#!/usr/bin/env python
# encoding: utf-8

# File        : hybrid.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Mar 23
#
# Description :

import awkward as ak
import numpy as np
import uproot

from Common import *
from Util import *

"""
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

"""


class GMTStubs(Module):
    def __init__(self, name="GMTStubs", prefix="stub_"):
        super().__init__(name)
        self.prefix = prefix
        self.__bookRate()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith(self.prefix):
                setattr(self, k.split(self.prefix)[-1], event[k])
        ## Follow the code from FwdMuonTranslator
        # self.pt = self.pt * GMT_LSB_Pt
        # ## Understand the phi conversion
        # locrad = calc_phi_loc_rad_from_int(self.phi)
        # self.phi = calc_phi_glob_rad_from_loc(locrad, self.sector)
        # theta = calc_theta_rad_from_int(self.eta)
        # self.eta =self.endcap*calc_eta_from_theta_rad(theta)
        super().__GetEvent__(event)

    def __bookRate(self):
        self.__bookobj()
        # self.BookEff()

    def __bookobj(self, name=None):
        objhist = {
            # "pt" :  Hist.new.Reg(100, 0, 100, name="pt").Double(),
            "nBX": Hist.new.Reg(10, -5, 5, name="BX").Double(),
            "nStubs": Hist.new.Reg(30, 0, 60, name="nStubs").Double(),
            "phi1": Hist.new.Reg(200, -500, 500, name="phi1").Double(),
            "phi2": Hist.new.Reg(200, -500, 500, name="phi2").Double(),
            "eta1": Hist.new.Reg(120, -120, 120, name="eta1").Double(),
            "eta2": Hist.new.Reg(120, -120, 120, name="eta2").Double(),
            "gphi1": Hist.new.Reg(200, -4, 4, name="gphi1").Double(),
            "gphi2": Hist.new.Reg(200, -4, 4, name="gphi2").Double(),
            "geta1": Hist.new.Reg(120, -4, 4, name="geta1").Double(),
            "geta2": Hist.new.Reg(120, -4, 4, name="geta2").Double(),
            "addr": Hist.new.Reg(500, 0, 10000, name="address").Double(),
            "etaregion": Hist.new.Reg(100, 0, 100, name="etaregion").Double(),
            "phiregion": Hist.new.Reg(100, 0, 100, name="phiregion").Double(),
            "depthregion": Hist.new.Reg(100, 0, 100, name="depthregion").Double(),
            "tflayer": Hist.new.Reg(100, 0, 100, name="tflayer").Double(),
            # "rate" : Hist.new.Reg(100, 0, 100, name="rate").Double(),
            # "qual" : Hist.new.Reg(16, 0, 16, name="qual").Double(),
        }
        self.h.update(objhist)

    def __fillRate(self):
        # self.h["pt"].fill(ak.ulatten(self.pt))
        # print("address ", self.addr)
        self.h["nStubs"].fill(ak.count(self.phi1, axis=1))
        self.h["nBX"].fill(ak.flatten(self.BX))
        self.h["addr"].fill(ak.flatten(self.addr))
        self.h["phi1"].fill(ak.flatten(self.phi1))
        self.h["phi2"].fill(ak.flatten(self.phi2))
        self.h["eta1"].fill(ak.flatten(self.eta1))
        self.h["eta2"].fill(ak.flatten(self.eta2))
        self.h["gphi1"].fill(ak.flatten(self.offphi1))
        self.h["gphi2"].fill(ak.flatten(self.offphi2))
        self.h["geta1"].fill(ak.flatten(self.offeta1))
        self.h["geta2"].fill(ak.flatten(self.offeta2))
        self.h["etaregion"].fill(ak.flatten(self.etaregion))
        self.h["phiregion"].fill(ak.flatten(self.phiregion))
        self.h["depthregion"].fill(ak.flatten(self.depthregion))
        self.h["tflayer"].fill(ak.flatten(self.tfLayer))
        # self.h["rate"].fill(ak.flatten(self.pt))

    # def BookEff(self):
    # self.h["effemu"] = Hist.new.Reg(100, 0, 100, name="rate").Double()
    # self.h["emu"] = Hist.new.Reg(100, 0, 100, name="rate").Double()
    # self.h["eff_eta_de"] =  Hist.new.Reg(90, -3, 3, name="eta;eta;y").Double()
    # self.h["eff_eta_nu"] =  Hist.new.Reg(90, -3, 3, name="eta;eta;y").Double()

    # def CalEff(self):
    # super().__CalDefaultEff__()

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()
        # self.CalEff()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
