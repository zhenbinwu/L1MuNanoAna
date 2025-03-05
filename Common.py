#!/usr/bin/env python
# encoding: utf-8

# File        : Common.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Apr 14
#
# Description :

import numpy as np

np.seterr(divide="ignore", invalid="ignore")
import math

import awkward as ak
import vector
from hist import Hist
from hist.intervals import clopper_pearson_interval, ratio_uncertainty

from Util import *

vector.register_awkward()

EMTFSiteMap = {
    0: "ME11",
    1: "ME12,3",
    2: "ME21,2",
    3: "ME31,2",
    4: "ME41,2",
    5: "RE12,3",
    6: "RE22,3",
    7: "RE31,2,3",
    8: "RE41,2,3",
    9: "GE11",
    10: "GE21",
    11: "ME0",
}


TFLayer = {
    0: 0,  # "ME11",
    1: 4,  # "ME12,3",
    2: 2,  # "ME21,2",
    3: 1,  # "ME31,2",
    4: 3,  # "ME41,2",
    5: 4,  # "RE12,3",
    6: 2,  # "RE22,3",
    7: 1,  # "RE31,2,3",
    8: 3,  # "RE41,2,3",
    9: -1,  # "GE11",
    10: -1,  # "GE21",
    11: -1,  # "ME0",
}

sysnum = ["DT", "CSC", "RPC", "GEM", "ME0"]
LHCnBunches = 2760
LHCFreq = 11.246  # kHz

phiLSB = 0.016666  # EMTF, From Patrick, 1/8 strip of MEx2/1
phiLSB2 = 1.0 / 360
## From the emulator, the conversion has 1/360 as resolution
thetaLSB = 1 / 3  # From TPUtils

LSB_EMTF_phi = 1.0 / 360
LSB_EMTF_theta = 1.0 / 3

LSB_GMTstub_phi = 0.00076660156 * 32
LSB_GMTstub_eta = 7.68334e-4 * 32

## GMT LSB
GMT_LSB_Pt = 0.03125
GMT_LSB_cor = 2 * math.pi / (1 << 13)

pTthresholds = [0, 3, 5, 10, 15, 20, 30]
plateaugap = 10  # Assuming 10 GeV to reach plateau
MuonEtamap = {  # [eta1, eta2, prompt_qual, displace_qual]
    "Barrel": [0, 0.83, 0, 0],
    "Overlap": [0.83, 1.24, 12, 12],
    "Endcap": [1.24, 2.4, 14, 15],
    "Endcap2": [1.24, 2.0, 14, 15],
}


def listLUT(ilist, target):
    LUT = ak.from_iter(ilist)
    cnts = ak.num(target)
    out = LUT[ak.flatten(target)]
    return ak.unflatten(out, cnts)


class Module:
    def __init__(self, name, isDisplaced=False, matchdR=0.3):
        # storing histogram
        self.h = {"Nevent": Hist.new.Reg(2, 0, 2, name="NO. of Events").Double()}
        self.folder = name
        self.p4 = None
        self.genmu = None
        self.effnames = []
        self.isDisplaced = isDisplaced
        self.matchdR = matchdR

    def __GetEvent__(self, event):
        if self.isDisplaced:
            self.genmu = self.__GetDispGenMuon__(event)
        else:
            self.genmu = self.__GetGenMuon__(event)

        self.__ConstructP4__()
        self.gen_pass, self.obj_pass = self.__MatchGenMax__(self.matchdR)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculating Efficiency ~~~~~
    def __GetTkMuon__(self, event):
        t = ak.zip(
            {
                "pt": event["L1gmtTkMuon_pt"],
                "phi": event["L1gmtTkMuon_phi"],
                "eta": event["L1gmtTkMuon_eta"],
                "Q": event["L1gmtTkMuon_charge"],
                "qual": event["L1gmtTkMuon_hwQual"],
                "idx": ak.local_index(event["L1gmtTkMuon_hwQual"]),
            }
        )
        return vector.Array(t)

    def __GetGenMuon__(self, event):
        parent = event["GenPart_pdgId"][event["GenPart_genPartIdxMother"]]
        sel = (
            (abs(event["GenPart_pdgId"]) == 13) & (event["GenPart_status"] == 1)
            # & (abs(event["GenPart_eta"]) < 2.4)
        )

        t = ak.zip(
            {
                "pt": event["GenPart_pt"][sel],
                "phi": event["GenPart_phi"][sel],
                "eta": event["GenPart_eta"][sel],
                "m": event["GenPart_mass"][sel],
                "vz": event["GenPart_vertZ"][sel],
                "dxy": event["GenPart_dXY"][sel],
                "lxy": event["GenPart_lXY"][sel],
                "orgphi": event["GenPart_phi"][sel],
                "orgeta": event["GenPart_eta"][sel],
                "parent": parent[sel],
                "idx": ak.local_index(event["GenPart_eta"])[sel],
                "mom": event["GenPart_genPartIdxMother"][sel],
            }
        )
        return vector.Array(t)

    def __GetGen__(self, event):
        t = ak.zip(
            {
                "pt": event["GenPart_pt"],
                "phi": event["GenPart_phi"],
                "eta": event["GenPart_eta"],
                "m": event["GenPart_mass"],
                "vz": event["GenPart_vertZ"],
                "id": event["GenPart_pdgId"],
                "dxy": event["GenPart_dXY"],
                "lxy": event["GenPart_lXY"],
                "mom": event["GenPart_genPartIdxMother"],
            }
        )
        return vector.Array(t)

    def __GetGenTau__(self, event):
        sel = (abs(event["GenPart_pdgId"]) == 15) & (abs(event["GenPart_eta"]) < 2.4)

        t = ak.zip(
            {
                "pt": event["GenPart_pt"][sel],
                "phi": event["GenPart_phi"][sel],
                "eta": event["GenPart_eta"][sel],
                "m": event["GenPart_mass"][sel],
                "vz": event["GenPart_vertZ"][sel],
                "dxy": event["GenPart_dXY"][sel],
                "lxy": event["GenPart_lXY"][sel],
                "idx": ak.local_index(event["GenPart_eta"])[sel],
                "mom": event["GenPart_genPartIdxMother"][sel],
            }
        )
        return vector.Array(t)

    def __GetDispGenMuon__(self, event):
        parent = event["GenPart_pdgId"][event["GenPart_genPartIdxMother"]]
        sel = (
            (abs(event["GenPart_pdgId"]) == 13)
            & (event["GenPart_status"] == 1)
            & (abs(event["GenPart_vertZ"]) < 500)
            & (abs(parent) == 6000113)
        )

        endcap_etaStar, endcap_phiStar = calc_etaphi_star_simple(
            event["GenPart_vertX"],
            event["GenPart_vertY"],
            event["GenPart_vertZ"],
            event["GenPart_eta"],
            event["GenPart_phi"],
        )

        barrel_etaStar = akgetSt2Eta(
            event["GenPart_lXY"],
            event["GenPart_vertZ"],
            event["GenPart_eta"],
        )
        barrel_phiStar = akgetSt2Phi(
            event["GenPart_vertX"],
            event["GenPart_vertY"],
            event["GenPart_phi"],
        )
        ## Pick the cross of MB2 and ME2 at eta of 1.2
        prop_etaStar = ak.where(
            abs(barrel_etaStar) <= 1.2, barrel_etaStar, endcap_etaStar
        )
        prop_phiStar = ak.where(
            abs(barrel_etaStar) <= 1.2, barrel_phiStar, endcap_phiStar
        )

        sel = sel & (abs(prop_etaStar) < 2.0)

        t = ak.zip(
            {
                "pt": event["GenPart_pt"][sel],
                "phi": prop_phiStar[sel],
                "eta": prop_etaStar[sel],
                "m": event["GenPart_mass"][sel],
                "vz": event["GenPart_vertZ"][sel],
                "dxy": event["GenPart_dXY"][sel],
                "lxy": event["GenPart_lXY"][sel],
                "orgphi": event["GenPart_phi"][sel],
                "orgeta": event["GenPart_eta"][sel],
            }
        )

        return vector.Array(t)

    def __ConstructP4__(self):
        if not hasattr(self, "pt"):
            print("Need to set object physics pt")
        if not hasattr(self, "eta"):
            print("Need to set object physics eta")
        if not hasattr(self, "phi"):
            print("Need to set object physics phi")

        t = ak.zip(
            {
                "pt": self.pt,
                "phi": self.phi,
                "eta": self.eta,
                "m": ak.zeros_like(self.pt),
            }
        )
        self.p4 = vector.Array(t)

    def __MatchGenMax__(self, match_dR=0.3):
        ref_test = ak.argcartesian({"gen": self.genmu, "L1": self.p4})
        gens, l1s = ak.unzip(ref_test)
        org_max = ak.argmax(self.genmu.pt, axis=-1, keepdims=True)
        dR = self.genmu[gens].deltaR(self.p4[l1s])
        pass_dR = dR < match_dR
        g_pass = gens[pass_dR]
        l_pass = l1s[pass_dR]
        ## Now need to check duplicate matches
        g_length = ak.run_lengths(g_pass)
        l_length = ak.run_lengths(l_pass)
        # g_max = ak.argmax(self.genmu.pt[g_pass], axis=-1, keepdims=True)
        return g_pass, l_pass

    def __bookEffReg__(self, name, nbins, xfirst, xlast, label=None):
        if name not in self.effnames:
            self.effnames.append((name, label))
        if name + "__den_" not in self.h:
            self.h["%s__den_" % name] = Hist.new.Reg(
                nbins, xfirst, xlast, name="% s__den_" % name, label=label
            ).Double()
        if name + "__num_" not in self.h:
            self.h["%s__num_" % name] = Hist.new.Reg(
                nbins, xfirst, xlast, name="% s__num_" % name, label=label
            ).Double()
        if name + "_eff" not in self.h:
            self.h["%s_eff" % name] = Hist.new.Reg(
                nbins, xfirst, xlast, name="% s_eff" % name, label=label
            ).Weight()

    def __bookEffVar__(self, name, edges, label=None):
        if name not in self.effnames:
            self.effnames.append((name, label))
        if name + "__den_" not in self.h:
            self.h["%s__den_" % name] = Hist.new.Var(
                edges, name="% s__den_" % name, label=label
            ).Double()
        if name + "__num_" not in self.h:
            self.h["%s__num_" % name] = Hist.new.Var(
                edges, name="% s__num_" % name, label=label
            ).Double()
        if name + "_eff" not in self.h:
            self.h["%s_eff" % name] = Hist.new.Var(
                edges, name="% s_eff" % name, label=label
            ).Weight()

    def __FillEff__(
        self, name, att, nbins, xfirst, xlast, label=None, gencut=None, objcut=None
    ):
        self.__bookEffReg__(name, nbins, xfirst, xlast, label)
        return self.__FillEffCal__(name, att, gencut, objcut)

    def __FillEffVar__(self, name, att, edges, label=None, gencut=None, objcut=None):
        self.__bookEffVar__(name, edges, label)
        return self.__FillEffCal__(name, att, gencut, objcut)

    def __FillEffPerObj__(
        self, name, obj, att, nbins, xfirst, xlast, label=None, dencut=None, numcut=None
    ):
        self.__bookEffReg__(name, nbins, xfirst, xlast, label)
        if dencut is None:
            dencut = ak.ones_like(obj, dtype=bool)
        if numcut is None:
            numcut = ak.ones_like(obj, dtype=bool)

        self.h["%s__den_" % name].fill(ak.flatten(getattr(obj[dencut], att)))
        self.h["%s__num_" % name].fill(ak.flatten(getattr(obj[numcut], att)))

    def __FillEffCal__(self, name, att, gencut=None, objcut=None):
        if gencut is None:
            gencut = ak.ones_like(self.genmu.pt, dtype=bool)
        if objcut is None:
            objcut = ak.ones_like(self.p4.pt, dtype=bool)

        ## First get the gen cut
        denominator_genidx = ak.local_index(self.genmu)[gencut]

        ## Now get the numerator index
        ## Get the matches index that pass gencut
        xs = ak.argcartesian({"obj": self.gen_pass, "cut": denominator_genidx})
        pass_gencut_idx = xs.obj[(self.gen_pass[xs.obj] == denominator_genidx[xs.cut])]
        newgen_pass = self.gen_pass[pass_gencut_idx]
        newobj_pass = self.obj_pass[pass_gencut_idx]

        # Get the obj index after cut
        objcut_idx = ak.local_index(self.p4)[objcut]
        ## Matching obj_pass with objcut_idx
        xs = ak.argcartesian({"obj": newobj_pass, "cut": objcut_idx})
        numerator_genidx = newgen_pass[
            xs.obj[(newobj_pass[xs.obj] == objcut_idx[xs.cut])]
        ]
        numerator_objidx = newobj_pass[
            xs.obj[(newobj_pass[xs.obj] == objcut_idx[xs.cut])]
        ]

        ## In case of duplicate matching
        numerator_unigenidx = GetUnique(numerator_genidx)

        self.h["%s__den_" % name].fill(
            ak.flatten(getattr(self.genmu[denominator_genidx], att))
        )
        self.h["%s__num_" % name].fill(
            ak.flatten(getattr(self.genmu[numerator_unigenidx], att))
        )
        return denominator_genidx, numerator_genidx, numerator_objidx

    def __CalDefaultEff__(self):
        for cut in pTthresholds:
            self.__FillEff__(
                "pt_%d" % cut,
                "pt",
                50,
                0,
                100,
                label="gen#mu p_{T}",
                objcut=self.pt > cut,
            )
            self.__FillEff__(
                "eta_pt_%d" % cut,
                "eta",
                180,
                -3,
                3,
                label="gen#mu #eta",
                gencut=self.genmu.pt > (10 + cut),
                objcut=self.pt > cut,
            )
            self.__FillEff__(
                "phi_pt_%d" % cut,
                "phi",
                200,
                -4,
                4,
                label="gen#mu #phi",
                gencut=self.genmu.pt > (10 + cut),
                objcut=self.pt > cut,
            )
            for region, etas in MuonEtamap.items():
                self.__FillEff__(
                    "%s_pt_%d" % (region, cut),
                    "pt",
                    50,
                    0,
                    100,
                    label="%s gen#mu p_{T}" % region,
                    gencut=(abs(self.genmu.eta) >= etas[0])
                    & (abs(self.genmu.eta) < etas[1]),
                    objcut=self.pt > cut,
                )
                self.__FillEff__(
                    "%s_phi_%d" % (region, cut),
                    "phi",
                    200,
                    -4,
                    4,
                    label="%s gen#mu #phi" % region,
                    gencut=(abs(self.genmu.eta) >= etas[0])
                    & (abs(self.genmu.eta) < etas[1]),
                    objcut=self.pt > cut,
                )

    def __SethbStation(self, event):
        ## Setting the hybrid stub stations
        isME11 = ((event.hit_emtf_chamber >= 0) & (event.hit_emtf_chamber <= 2)) | (
            (event.hit_emtf_chamber >= 9) & (event.hit_emtf_chamber <= 11)
        )
        isME0 = (event.hit_emtf_chamber >= 108) & (event.hit_emtf_chamber <= 114)
        isGE11 = ((event.hit_emtf_chamber >= 54) & (event.hit_emtf_chamber <= 56)) | (
            (event.hit_emtf_chamber >= 63) & (event.hit_emtf_chamber <= 11)
        )
        ## From the hybrid stub plot, only ME0/GE11 is station 1
        istation1 = isME0 | isGE11
        self.hb_station = event.hit_station + 1
        self.hb_station = ak.where(istation1, 1, self.hb_station)
        ## Set the Hybrid Stub Layers according to the plot
        self.hb_layer = self.hb_station

        ## Get the TFLayer, used in the GMT emulator
        self.hb_tflayer = listLUT(TFLayer.values(), event.hit_emtf_site)

        ## For ME11, eta < 2.0 will be station 1
        ME11_Stat1 = isME11 & (
            (abs(event.hit_glob_theta) < 15.4)
            | (abs(event.hit_glob_theta) > 180 - 15.4)
        )
        self.hb_layer = ak.where(ME11_Stat1, 1, self.hb_layer)
        ## For GE11, eta > 2.0 will be station 2
        GE11_Stat2 = isGE11 & (
            (abs(event.hit_glob_theta) > 15.4)
            & (abs(event.hit_glob_theta) < 180 - 15.4)
        )
        self.hb_layer = ak.where(GE11_Stat2, 2, self.hb_layer)
        # self.hit_tflayer = self.hb_layer

    def run(self, event):
        self.h["Nevent"].fill(event.event > 0)
        self.__GetEvent__(event)

    def endrun(self, outfile, nTotal=0):
        for eff, label in self.effnames:
            self.ConvertEff(eff, label)
        orgkeys = list(self.h.keys())
        for k in orgkeys:
            if "rate" in k:
                self.h["%s_scaled" % k] = self.ConvertRate(self.h[k], nTotal)
                self.h.pop(k)
        for k in self.h.keys():
            outfile["%s/%s" % (self.folder, k)] = self.h[k]

    def ConvertEff(self, name, label):
        print(name, label)
        effname = name + "_eff"
        num = name + "__num_"
        den = name + "__den_"
        values = np.true_divide(
            self.h[num].values(),
            self.h[den].values(),
            out=np.zeros_like(self.h[num].values()),
            where=self.h[den].values() != 0,
        )
        variances = np.zeros_like(values)
        # variances = ratio_uncertainty(
        #     num=self.h[num].values(),
        #     denom=self.h[den].values(),
        #     uncertainty_type="efficiency",
        # )[0]
        self.h[effname][...] = np.stack([values, variances], axis=-1)
        # self.h.pop(num)
        # self.h.pop(den)

    def ConvertRate(self, hist, nZB=0):
        rethist = hist.copy()
        if nZB == 0:
            return hist
        content = hist.values()
        newcontent = np.flip(np.cumsum(np.flip(content))) * LHCnBunches * LHCFreq / nZB
        rethist[...] = newcontent
        return rethist
