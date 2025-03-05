#!/usr/bin/env python3.11
# encoding: utf-8

# File        : Tau23Mus.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2024 Oct 15
#
# Description :

import pprint
from collections import defaultdict
from math import e
from os import killpg, sendfile
from re import findall

import awkward as ak
import numpy as np

from Common import *


class TauModules(Module):
    def __init__(self, name="Tau23Mu", prefix="tau23mu_", matchdR=0.3):
        super().__init__(name, matchdR)
        self.prefix = prefix
        self.__bookobj()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith(self.prefix):
                setattr(self, k.split(self.prefix)[-1], event[k])
        self.mass = np.sqrt(2 * self.hwMass) * GMT_LSB_Pt * (1 << 2)
        self.genmu = super().__GetGenMuon__(event)
        self.gentau = super().__GetGenTau__(event)
        self.tkmu = super().__GetTkMuon__(event)
        self.tktau = ak.zip(
            {
                "mass": self.mass,
                "m1": self.mu1,
                "m2": self.mu2,
                "m3": self.mu3,
            }
        )

    def __bookobj(self, name=None):
        objhist = {
            "hwmass": Hist.new.Reg(300, 0, 300, name="hwmass").Double(),
            "mass": Hist.new.Reg(100, 0, 10, name="mass").Double(),
            "mu1": Hist.new.Reg(10, 0, 10, name="mu1").Double(),
            "mu2": Hist.new.Reg(10, 0, 10, name="mu2").Double(),
            "mu3": Hist.new.Reg(10, 0, 10, name="mu3").Double(),
            "tktau_rate": Hist.new.Reg(100, 0, 100, name="rate").Double(),
        }
        gentauhists = {
            "gentau_nMuonAccept": Hist.new.Reg(
                4, 0, 4, name="gentau_nMuonAccept"
            ).Double(),
            "gentau_accepted_pt": Hist.new.Reg(
                200, 0, 20, name="accepted gentau pt"
            ).Double(),
            "gentau_accepted_eta": Hist.new.Reg(
                200, -4, 4, name="accepted gentau eta"
            ).Double(),
            "gentau_accepted_phi": Hist.new.Reg(
                200, -4, 4, name="accepted gentau phi"
            ).Double(),
            "gentau_matched_pt": Hist.new.Reg(
                200, 0, 20, name="matched gentau pt"
            ).Double(),
            "gentau_matched_eta": Hist.new.Reg(
                200, -4, 4, name="matched gentau eta"
            ).Double(),
            "gentau_matched_phi": Hist.new.Reg(
                200, -4, 4, name="matched gentau phi"
            ).Double(),
            "gentau_missingmuon_pt": Hist.new.Reg(
                200, 0, 4, name="missing muon pt"
            ).Double(),
            "gentau_missingmuon_eta": Hist.new.Reg(
                200, -4, 4, name="missing muon eta"
            ).Double(),
            "gentau_missingmuon_pteta": Hist.new.Reg(200, 0, 4, name="missing muon pt")
            .Reg(200, -4, 4, name="missing muon eta")
            .Double(),
        }
        self.h.update(objhist)
        self.h.update(gentauhists)

    def __fillobj(self):
        self.h["hwmass"].fill(ak.flatten(self.hwMass))
        self.h["mass"].fill(ak.flatten(self.mass))
        self.h["mu1"].fill(ak.flatten(self.mu1))
        self.h["mu2"].fill(ak.flatten(self.mu2))
        self.h["mu3"].fill(ak.flatten(self.mu3))
        self.h["tktau_rate"].fill_flattened(ak.drop_none(ak.min(self.mass, axis=1)))

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillobj()
        self.caleffall()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)

    # ┌──────────────────────────────────────────────────────────┐
    # │ # There are several steps to calculate the efficiency    │
    # │ of the Tau                                               │
    # └──────────────────────────────────────────────────────────┘
    #  1. Build up the correlation between objects
    #  Macthing gentau with gen muon
    def __ConnectGenTauGenMuall(self):
        cart = ak.argcartesian(
            {"gentau": self.gentau.idx, "genmu": self.genmu.mom}, axis=1
        )
        ltaus, lmus = ak.unzip(cart)
        sel = self.gentau[ltaus].idx == self.genmu[lmus].mom
        remap = ak.zip([ltaus[sel], lmus[sel]])
        return remap

    # Getting the matching genmu and tkmu
    def __ConnectGenTauGenMu(self, i):
        gentau = self.gentau[i]
        genmu = self.genmu[i]
        cart = ak.argcartesian({"gentau": gentau.idx, "genmu": genmu.mom}, axis=0)
        ltaus, lmus = ak.unzip(cart)
        sel = gentau[ltaus].idx == genmu[lmus].mom
        remap = {
            key: [v for k, v in zip(ltaus[sel], lmus[sel]) if k == key]
            for key in set(ltaus[sel])
        }
        print("gentau Genmu matching ", remap)
        for k, v in remap.items():
            print(k, v)
            print((genmu[v].pt > 2) & (abs(genmu[v].eta) < 2.5))
            for i in v:
                print(i, genmu[i].pt, genmu[i].eta)
        return remap

    def __ConnectGenMuTkMu(self, i, match_dR=0.3):
        tkmu = self.tkmu[i]
        genmu = self.genmu[i]
        cart = ak.argcartesian({"tkmu": tkmu.pt, "genmu": genmu.pt}, axis=0)
        ltks, lgens = ak.unzip(cart)
        dR = genmu[lgens].deltaR(tkmu[ltks])
        pass_dR = dR < match_dR
        gMuMap = {
            key: [v for k, v in zip(lgens[pass_dR], ltks[pass_dR]) if k == key]
            for key in set(lgens[pass_dR])
        }
        matched = {}

        for gmu, tks in gMuMap.items():
            dRs = genmu[gmu].deltaR(tkmu[tks])
            matched[gmu] = tks[ak.argmin(dRs)]
        return matched

    def __ConnectGenTauTkTau(self, i):
        tktau = self.tktau[i]
        # gentau1 = self.genmu[self.tkmu[self.tktau.mu1]._genmuIdx]
        # gentau2 = self.genmu[self.tkmu[self.tktau.mu2]._genmuIdx]
        # gentau3 = self.,genmu[self.tkmu[self.tktau.mu3]._genmuIdx]

    def caleffall(self):
        # Get the index of gen tau with 3 muon matched
        gentauaccpetd = self.GenTauStudy()
        # boolen of Event has accepted gen tau, event based
        gentauevt = ak.count(gentauaccpetd, axis=1) > 0
        # boolen of Event has tk tau, event based
        tktauevt = ak.count(self.hwMass, axis=1) > 0
        numaccepted = tktauevt & gentauevt
        gentaumatched = ak.mask(gentauaccpetd, numaccepted)

        self.__FillEffPerObj__(
            "gentau_eff_pt",
            self.gentau,
            "pt",
            100,
            0,
            20,
            "pt_eff",
            gentauaccpetd,
            gentaumatched,
        )
        self.__FillEffPerObj__(
            "gentau_eff_mass",
            self.gentau,
            "m",
            100,
            0,
            4,
            "mass_eff",
            gentauaccpetd,
            gentaumatched,
        )
        self.__FillEffPerObj__(
            "gentau_eff_eta",
            self.gentau,
            "eta",
            100,
            -4,
            4,
            "eta_eff",
            gentauaccpetd,
            gentaumatched,
        )
        self.__FillEffPerObj__(
            "gentau_eff_phi",
            self.gentau,
            "phi",
            100,
            -4,
            4,
            "phi_eff",
            gentauaccpetd,
            gentaumatched,
        )

    def GenTauStudy(self):
        # study of gen tau
        gentaugenmu = self.__ConnectGenTauGenMuall()
        lt, lm = ak.unzip(gentaugenmu)
        # gen muon that falled with trakcer muon acceptance
        muappt = (self.genmu[lm].pt > 2) & (abs(self.genmu[lm].eta) < 2.5)
        self.h["gentau_nMuonAccept"].fill_flattened(ak.run_lengths(lt[muappt]))
        self.h["gentau_missingmuon_pt"].fill(ak.flatten(self.genmu[lm[~muappt]].pt))
        self.h["gentau_missingmuon_eta"].fill(ak.flatten(self.genmu[lm[~muappt]].eta))
        self.h["gentau_missingmuon_pteta"].fill(
            ak.flatten(self.genmu[lm[~muappt]].pt),
            ak.flatten(self.genmu[lm[~muappt]].eta),
        )
        ## Only accept the gen tau with 3 muon matched
        lt_accpeted = lt[ak.run_lengths(lt[muappt]) == 3]
        self.h["gentau_accepted_pt"].fill_flattened(self.gentau[lt_accpeted].pt)
        self.h["gentau_accepted_eta"].fill_flattened(self.gentau[lt_accpeted].eta)
        self.h["gentau_accepted_phi"].fill_flattened(self.gentau[lt_accpeted].phi)
        return lt_accpeted

    def caleff(event):
        ## Object self matching by event, otherwise use too much memory
        for i, _ in enumerate(self.gentau):
            gentaugenmu = self.__ConnectGenTauGenMu(i)
            genmutkmu = self.__ConnectGenMuTkMu(i)
            print(i, "gentau", gentaugenmu, "genmu", genmutkmu)
            if i > 10:
                break

    def calTaueff(self):
        self.__bookEffReg__("taueff_mass", "mass", 100, 0, 100, label="tau mass")
        self.__FillEffPerObj__(
            "taueff_mass",
            self.gentau,
            "mass",
            len(self.gentau) == 1,
            len(self.tktau) == 1,
        )
