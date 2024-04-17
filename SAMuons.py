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
            "dxy" :  Hist.new.Reg(100, 0, 100, name="dxy").Double(),
            "gendxy" :  Hist.new.Reg(100, 0, 100, name="gendxy").Double(),
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
        self.h["pt"].fill(ak.flatten(self.pt))
        self.h["dxy"].fill(ak.flatten(self.d0))
        self.h["gendxy"].fill(ak.flatten(self.genmu.dxy))
        self.h["phi"].fill(ak.flatten(self.phi))
        self.h["eta"].fill(ak.flatten(self.eta))
        self.h["rate"].fill(ak.drop_none(ak.max(self.pt, axis=1)))
        self.h["Endcap_phi"].fill(ak.flatten(self.phi[abs(self.eta)>1.2]))
        self.h["Endcap_phi2"].fill(ak.flatten(self.hwPhi[abs(self.eta)>1.2] * GMT_LSB_cor))
        for i in range(12):
            self.h["rate_qual%d" % i].fill(ak.drop_none(ak.max(self.pt[self.hwQual >= i], axis=1 )))
            for region, etas in MuonEtamap.items():
                sel = (self.hwQual >= i) & (abs(self.eta)>= etas[0]) & (abs(self.eta) < etas[1])
                self.h["%s_rate_qual%d" % (region, i)].fill(ak.drop_none(ak.max(self.pt[sel], axis=1)))
        # self.h["detadphi_dxy4"].fill(ak.flatten(self.eta))


    def BookEff(self):
        self.h["effemu"] = Hist.new.Reg(100, 0, 100, name="rate").Double()
        self.h["emu"] = Hist.new.Reg(100, 0, 100, name="rate").Double()
        self.h["eff_eta_de"] =  Hist.new.Reg(90, -3, 3, name="eta;eta;y").Double()
        self.h["eff_eta_nu"] =  Hist.new.Reg(90, -3, 3, name="eta;eta;y").Double()

    def CalEff(self):
        super().__CalDefaultEff__()
        for cut in pTthresholds:
            self.__FillEff__("KMTF_dxy_pt%d" % (cut), "dxy", 20, 0, 100, label="gen KMTF #mu d_{xy} [cm]",
                             gencut = getKMTFAcceptance(self.genmu.lxy,
                                                        self.genmu.vz,
                                                        self.genmu.orgeta) &  (self.genmu.pt > 10+cut),
                             objcut = (self.pt > cut)
                            )
            self.__FillEff__("KMTF_lxy_pt%d" % (cut), "lxy", 20, 0, 100, label="gen KMTF #mu l_{xy} [cm]",
                             gencut = getKMTFAcceptance(self.genmu.lxy,
                                                        self.genmu.vz,
                                                        self.genmu.orgeta) &  (self.genmu.pt > 10+cut),
                             objcut = (self.pt > cut)
                            )

            for qcut in [0, 1, 2, 3, 4, 8, 12, 14, 15]:
                self.__FillEff__("dxy_pt%d_qual%d" % (cut, qcut), "dxy", 25, 0, 100, 
                                 label="gen#mu d_{xy} [cm]",
                                 gencut = (abs(self.genmu.eta)<2.0), 
                                 objcut = (self.pt > cut) & (self.hwQual >= qcut)
                                )

                for region, etas in MuonEtamap.items():
                    self.__FillEff__("%s_pt%d_qual%d" % (region, cut, qcut), "pt", 25, 0, 100, label="%s gen#mu p_{T}" % region,
                                     gencut = (abs(self.genmu.eta)>= etas[0]) & (abs(self.genmu.eta) < etas[1]),
                                     objcut = (self.pt > cut) & (self.hwQual >= qcut) & (abs(self.eta)>= etas[0]) & (abs(self.eta) < etas[1])
                                    )
                    self.__FillEff__("%s_phi%d_qual%d" % (region, cut, qcut), "phi", 200, -4, 4, 
                                     label="%s gen#mu #phi" % region, 
                                     gencut = (abs(self.genmu.eta)>= etas[0]) & (abs(self.genmu.eta) < etas[1]),
                                     objcut = (self.pt > cut) & (self.hwQual >= qcut)
                                    )
                    self.__FillEff__("%s_dxy_pt%d_qual%d" % (region, cut, qcut), "dxy", 20, 0, 100, label="gen#mu d_{xy} [cm]",
                                     gencut = (abs(self.genmu.eta)>= etas[0]) & (abs(self.genmu.eta) < etas[1]) & (self.genmu.pt > 10+cut),
                                     objcut = (self.pt > cut) & (self.hwQual >= qcut) & (abs(self.eta)>= etas[0]) & (abs(self.eta) < etas[1])
                                    )
                    self.__FillEff__("%s_lxy_pt%d_qual%d" % (region, cut, qcut), "lxy", 20, 0, 100, label="gen#mu l_{xy} [cm]",
                                     gencut = (abs(self.genmu.eta)>= etas[0]) & (abs(self.genmu.eta) < etas[1]) & (self.genmu.pt > 10+cut),
                                     objcut = (self.pt > cut) & (self.hwQual >= qcut) & (abs(self.eta)>= etas[0]) & (abs(self.eta) < etas[1])
                                    )

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()
        self.CalEff()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
