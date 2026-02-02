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
    def __init__(self, name="TkMuons", prefix="tkmu_"):
        self.prefix = prefix
        super().__init__(name)
        self.__bookRate()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith(self.prefix):
                setattr(self, k.split(self.prefix)[-1], event[k])
        super().__GetEvent__(event)

    def __bookRate(self):
        self.h.update(
            {
                "tkmu_rate": Hist.new.Reg(100, 0, 100, name="rate").Double(),
                "tkmu_qual": Hist.new.Reg(1024, 0, 1024, name="qual").Double(),
                "tkmu_isosum": Hist.new.Reg(1024, 0, 1024, name="isosum").Double(),
                "tkmu_pt": Hist.new.Reg(30, 0, 30, name="pt").Double(),
                "tkmu_eta": Hist.new.Reg(60, -3, 3, name="eta").Double(),
                "tkmu_phi": Hist.new.Reg(80, -4, 4, name="phi").Double(),
            }
        )

    def CalEff(self):
        super().__CalDefaultEff__()

    def __fillRate(self):
        self.h["tkmu_rate"].fill(ak.flatten(self.pt))
        self.h["tkmu_qual"].fill(ak.flatten(self.hwQual))
        self.h["tkmu_isosum"].fill(ak.flatten(self.hwIso))
        self.h["tkmu_pt"].fill(ak.flatten(self.pt))
        self.h["tkmu_eta"].fill(ak.flatten(self.eta))
        self.h["tkmu_phi"].fill(ak.flatten(self.phi))

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()
        self.findduplicateTracks()
        self.CalEff()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)

    def findduplicateTracks(self):
        zipped = ak.zip(
            {"eta": self.hwEta, "phi": self.hwPhi, "pt": self.hwPt}, with_name="trk"
        )

        def point_equal(left, right):
            return np.logical_and(
                np.logical_and(left.eta == right.eta, left.phi == right.phi),
                left.pt == right.pt,
            )

        ak.behavior[np.equal, "trk", "trk"] = point_equal
        paircom = ak.combinations(zipped, 2, axis=1, fields=["0", "1"])
        hasdup = ak.any(paircom["0"] == paircom["1"], axis=1)
        print(len(hasdup))
        dup = ak.local_index(zipped, axis=0)[hasdup]
        print(len(dup))
        dup.show()
        # print()


# zipped[dup]
# zipped[dup][0].show()

# t =ak.cartesian({"1":self.hwEta, "2": self.hwEta})
# print(t)
# df= ak.to_dataframe(zipped)
# print(df)
# print(df.columns)
# print(df.to_flat_index())
# duponly = df.duplicated(subset=["entry", "0", "1","2"])
# # t = df.groupby(level=0).duplicated()
# t = df.groupby(level='subentry')
# print(t.first())
# dff = df[duponly]
# print(dff)
# cnt = df.reset_index(level=1).index.value_counts()
# print(cnt)
# self.h["trk_dup_bits"].fill(cnt.values)

# df= ak.to_dataframe(ak.zip([self.l1t_trk_hwPt, self.l1t_trk_phiI,
# self.l1t_trk_etaI]))
# duponly = df.duplicated()
# dff = df[duponly]
# cnt = dff.reset_index(level=1).index.value_counts()
# self.h["trk_dup_conv"].fill(cnt.values)
# df= ak.to_dataframe(ak.zip([self.l1t_trk_pt, self.l1t_trk_phi,
# self.l1t_trk_eta]))
# duponly = df.duplicated()
# dff = df[duponly]
# cnt = dff.reset_index(level=1).index.value_counts()
# self.h["trk_dup_float"].fill(cnt.values)
