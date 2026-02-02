#!/usr/bin/env python
# encoding: utf-8

# File        : hits.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Mar 11
#
# Description :

import uproot
import numpy as np
import pandas as pd
import awkward as ak
from hist import Hist
import hist
from pprint import pprint
from Common import *
from itertools import chain
from bidict import bidict

""" todo

"""

# Convert Segments into host/chamber
ch_ids = [
    # ME0 - This should change in the emu (only 5 chambers)
    [114, 108, 109, 110, 111, 112, 113],
    [99, 54, 55, 56, 63, 64, 65],  # GE11
    [45, 0, 1, 2, 9, 10, 11],  # ME11
    [46, 3, 4, 5, 12, 13, 14],  # ME12
    [100, 57, 58, 59, 66, 67, 68],  # RE12
    [102, 72, 73, 74, -1, -1, -1],  # GE21
    [103, 75, 76, 77, 78, 79, 80],  # RE22
    [48, 18, 19, 20, -1, -1, -1],  # ME21
    [49, 21, 22, 23, 24, 25, 26],  # ME22
    [50, 27, 28, 29, -1, -1, -1],  # ME31
    [51, 30, 31, 32, 33, 34, 35],  # ME32
    [104, 81, 82, 83, -1, -1, -1],  # RE31
    [105, 84, 85, 86, 87, 88, 89],  # RE32
    [52, 36, 37, 38, -1, -1, -1],  # ME41
    [53, 39, 40, 41, 42, 43, 44],  # ME42
    [106, 90, 91, 92, -1, -1, -1],  # RE41
    [107, 93, 94, 95, 96, 97, 98],  # RE42
    [47, 6, 7, 8, 15, 16, 17],  # ME13
    [101, 60, 61, 62, 69, 70, 71],  # RE13 - never actually used
]

matchCSC_ = {
    # 2: [0, 1],  # ME11 : ME0, GE11
    2: 0,
    2: 1,  # ME11 : ME0, GE11
    3: 4,  # ME12:RE12
    17: 18,  # ME13: RE13
    7: 5,  # ME21: GE21
    8: 6,  # ME22: RE22
    9: 11,  # ME31: RE31
    10: 12,  # ME32: RE32
    13: 15,  # ME41: RE41
    14: 16,  # ME42: RE42
}
matchCSC = matchCSC_.copy()
for i, v in matchCSC_.items():
    matchCSC[v] = i


""" Variables in L1Nano (Removed leading EMTF)
hit_chamber
hit_cscFR
hit_cscId
hit_bend
hit_chamber
hit_host
hit_phi
hit_emtf_qual1
hit_emtf_qual2
hit_emtf_segment
hit_emtf_site
hit_emtf_theta1
hit_emtf_theta2
hit_emtf_time
hit_emtf_timezones
hit_emtf_zones
hit_endcap
hit_fneigh
hit_fsub
hit_fvalid
hit_globPerp
hit_globPhi
hit_globTheta
hit_globTime
hit_globZ
hit_id
hit_layer
hit_ring
hit_roll
hit_sector
hit_station
hit_subsector
hit_subsystem

"""


class EMTFHits(Module):
    def __init__(self, name="EMTFHits"):
        super().__init__(name)
        ## Booking all the histograms
        self.__bookSecCnt()

    def __GetEvent__(self, event):
        super().__GetEvent__(event)
        super().__SethbStation__(event)
        for k in dir(event):
            if k.startswith("emtfhit_"):
                setattr(self, k.replace("emtf", "", 1), event[k])

    def run(self, event):
        super().run(event)
        self.MatchingChamber()
        # ## Basic understanding of the EMTFHits
        self.plotSecCnt()

    def MatchingChamber(self):
        chm = ak.zip(
            {
                "chamber": self.hit_chamber,
                "host": self.hit_host,
                "subsys": self.hit_subsystem,
                "sys": self.hb_station,
                "layer": self.hb_layer,
            }
        )
        ids = list(chain.from_iterable(ch_ids))
        for eidx, e in enumerate(self.hit_chamber):
            matching = {}
            print(e)
            for j, i in enumerate(e):  ## Loop over the hits in the event e:
                if i not in matching.keys():
                    matching[i] = []
                idspos = ids.index(i)
                if idspos // 7 == 0:  ## Skip ME0 for now
                    continue
                print(i, idspos // 7, idspos % 7)
                ## Get the matching chamber
                matchsys = matchCSC[idspos // 7]
                ch = ch_ids[matchsys][idspos % 7]
                if ch in matching.keys():
                    matching[ch].append(i)
            for k, v in matching.items():
                print(k, v)

            print("maching event", matching)

    def __bookSecCnt(self):
        ## Trying to understand the hits chamber vs hit vs segment
        self.h.update(
            {
                "emtfhit_chamber": Hist.new.Reg(
                    128, 0, 128, name="Chamber ID"
                ).Double(),
                "hit_chamber": Hist.new.Reg(
                    128, 0, 128, name="hit Chamber ID"
                ).Double(),
                "emtfhit_site": Hist.new.Reg(128, 0, 128, name="Site").Double(),
                "emtfhit_host": Hist.new.Reg(128, 0, 128, name="Site").Double(),
                "emtfhit_segment": Hist.new.Reg(128, 0, 128, name="Segment").Double(),
                "emtfhit_ring": Hist.new.Reg(10, 0, 10, name="ring").Double(),
                "emtfhit_station": Hist.new.Reg(10, 0, 10, name="station").Double(),
                "ME0_theta": Hist.new.Reg(128, 0, 128, name="theta").Double(),
            }
        )
        self.h.update(
            {
                "CSC_Bending": Hist.new.Reg(
                    20, -10, 10, name="Bending angle of CSC;Bending;CSC Hits"
                ).Double(),
                "NStation": Hist.new.Reg(
                    10, 0, 10, name="NO of EMTF Hit Station;X;Y"
                ).Double(),
                "NHBLayer": Hist.new.Reg(
                    10, 0, 10, name="NO of Hybrid Stub Layer"
                ).Double(),
                "NTFLayer": Hist.new.Reg(10, 0, 10, name="NO of TF Layer").Double(),
                "seccnt": Hist.new.Reg(20, 0, 20, name="NO. of hits / Sector").Double(),
                "stationcnt": Hist.new.Reg(
                    20, 0, 20, name="NO. of hits / Station"
                ).Double(),
                "CSC_nStub": Hist.new.Reg(50, 0, 50, name="NO. of CSC stubs ").Double(),
                "RPC_nStub": Hist.new.Reg(50, 0, 50, name="NO. of RPC stubs ").Double(),
                "CSCovel_nStub": Hist.new.Reg(
                    50, 0, 50, name="NO. of CSC stubs with overlap "
                ).Double(),
                "RPCovel_nStub": Hist.new.Reg(
                    50, 0, 50, name="NO. of RPC stubs with overlap "
                ).Double(),
            }
        )
        for i in range(1, len(sysnum)):
            self.h["seccnt_type%d" % i] = Hist.new.Reg(
                20, 0, 20, name="NO. of %s hits / Sector" % sysnum[i]
            ).Double()

        for i in range(1, 6):
            self.h["stationcnt%d" % i] = Hist.new.Reg(
                20, 0, 20, name="NO. of hits / Station %d" % i
            ).Double()
            for j in range(0, len(sysnum)):
                self.h["seccnt_type%d_sta%d" % (j, i)] = Hist.new.Reg(
                    20, 0, 20, name="NO. of %s hits / Station %i" % (sysnum[j], i)
                ).Double()
        for i in EMTFSiteMap.keys():
            self.h["phidist_%s" % EMTFSiteMap[i]] = Hist.new.Reg(
                60, -30, 30, name="phi vs offline phi"
            ).Double()

        for i in EMTFSiteMap.keys():
            self.h["thetadist_%s" % EMTFSiteMap[i]] = Hist.new.Reg(
                400, 0, 200, name="theta per site"
            ).Double()
            self.h["theta1dist_%s" % EMTFSiteMap[i]] = Hist.new.Reg(
                200, 0, 200, name="theta1 per site"
            ).Double()
            self.h["theta2dist_%s" % EMTFSiteMap[i]] = Hist.new.Reg(
                200, 0, 200, name="theta2 per site"
            ).Double()
            self.h["etadist_%s" % EMTFSiteMap[i]] = Hist.new.Reg(
                400, -5, 5, name="eta per site"
            ).Double()

        for i in range(0, 19):
            self.h["theta1dist_host%d" % i] = Hist.new.Reg(
                200, 0, 200, name="theta1 of host %d" % i
            ).Double()
            self.h["theta2dist_host%d" % i] = Hist.new.Reg(
                200, 0, 200, name="theta2 of host %d" % i
            ).Double()
        self.h["station_map"] = (
            Hist.new.Reg(250, 500, 1000, name="x", label="Z")
            .Reg(180, 0, 180, name="y", label="theta")
            .Reg(10, 0, 10, name="z", label="Layer")
            .Double()
        )
        self.h["hit_thetaVsLayer"] = (
            Hist.new.Reg(360, 0, 360, name="x", label="theta")
            .Reg(5, 0.5, 5.5, name="y", label="TFLayer")
            .Double()
        )
        self.h["hit_siteVsBend"] = (
            Hist.new.Reg(12, 0, 12, name="x", label="site")
            .Reg(240, 120, -120, name="y", label="bend")
            .Double()
        )

    def __bookExtraHits(self):
        self.h.update(
            {
                "extra_cnt": Hist.new.Reg(
                    10, 0, 10, name="NO. of extra hits/ Sector"
                ).Double(),
                "extra_station": Hist.new.Reg(
                    20, 0, 20, name="NO. of hits / Station"
                ).Double(),
            }
        )

    def stubID(self):
        df = ak.to_dataframe(
            {
                "chamber": self.hit_chamber,
                "segment": self.hit_segment,
                "tflayer": self.hit_station,
            }
        )
        ## Get min qual per event/track/tf
        # g = df.groupby(["tflayer"])['chamber'].unique()
        # print("stubID", self.hit_emtf_chamber, self.hit_tflayer)
        # for i in range(1, 6):
        # print(i, len(g[i]), sorted(g[i]))

    def plotTFLayer(self):
        self.h["station_map"].fill(
            x=ak.flatten(abs(self.hit_glob_z)),
            y=ak.flatten(self.hit_glob_theta),
            z=ak.flatten(self.hb_layer),
        )
        self.h["hit_thetaVsLayer"].fill(
            x=ak.flatten(self.hit_glob_theta), y=ak.flatten(self.hb_layer)
        )
        self.h["hit_siteVsBend"].fill(
            x=ak.flatten(self.hit_site), y=ak.flatten(self.hit_bend)
        )
        self.h["CSC_Bending"].fill(ak.flatten(self.hit_bend[self.hit_site < 5]))

    def sortsplit(self, input, variable):
        sorted = input[ak.argsort(input[variable])]
        output = ak.unflatten(
            ak.flatten(sorted), ak.flatten(ak.run_lengths(sorted[variable]))
        )
        return output

    def plotSecCnt(self):
        ### Understand the CSC hits
        self.h["hit_chamber"].fill(ak.flatten(self.hit_chamber))
        self.h["emtfhit_chamber"].fill(ak.flatten(self.hit_chamber))
        self.h["emtfhit_site"].fill(ak.flatten(self.hit_site))
        self.h["emtfhit_host"].fill(ak.flatten(self.hit_host))
        self.h["emtfhit_segment"].fill(ak.flatten(self.hit_segment))
        self.h["emtfhit_ring"].fill(ak.flatten(self.hit_ring))
        self.h["emtfhit_station"].fill(ak.flatten(self.hit_station))
        ### Plot per sector/sation
        x = ak.zip({"sec": self.hit_endcap * self.hit_sector, "sta": self.hit_station})
        sorted = x[ak.argsort(x.sec)]
        cnt = ak.run_lengths(sorted.sta)
        self.h["stationcnt"].fill(ak.flatten(cnt))

        ### Create a large Zip
        seczip = ak.zip(
            {
                "sector": self.hit_endcap * self.hit_sector,
                "hitstation": self.hit_station,
                "hblayer": self.hb_station,
                "tflayer": self.hb_tflayer,
                "subsys": self.hit_subsystem,
                # "site": self.hit_site,
                # "host": self.hit_host,
            }
        )
        ### Sorted per sector
        sortsec = self.sortsplit(seczip, "sector")
        ### Plot subsystem per sector
        cnt = ak.run_lengths(sortsec.sector)
        self.h["seccnt"].fill(ak.flatten(cnt))
        sec_subsys = sortsec[ak.argsort(sortsec.subsys)]
        for i in range(1, len(sysnum)):
            self.h["seccnt_type%d" % i].fill(
                ak.flatten(ak.run_lengths(sec_subsys.sector[sec_subsys.subsys == i]))
            )
        ### Plot per station/sector
        # for i in range(1, 6)
        # stats = x.sec[x.station == i]
        # cnt = ak.flatten(ak.run_lengths(stats))
        # self.h["stationcnt%d"% i].fill(cnt)
        # for j in range(1, len(sysnum)):
        # syss = x.sec[ (x.station == i) & (x.sys == j) ]
        # cnt =ak.flatten(ak.run_lengths(syss))
        # self.h["seccnt_type%d_sta%d" %( j , i) ].fill(cnt)

    def StudyResolution(self):
        secedge = 15 + self.hit_endcap * 60
        secedge = ak.where(secedge > 180, secedge - 360, secedge)
        phidiff = self.hit_phi * phiLSB - self.hit_glob_phi
        for i in EMTFSiteMap.keys():
            self.h["phidist_%s" % EMTFSiteMap[i]].fill(
                ak.flatten(phidiff[self.hit_site == i])
            )
            self.h["thetadist_%s" % EMTFSiteMap[i]].fill(
                ak.flatten(self.hit_glob_theta[self.hit_site == i])
            )
            self.h["theta1dist_%s" % EMTFSiteMap[i]].fill(
                ak.flatten(self.hit_theta1[self.hit_site == i])
            )
            self.h["theta2dist_%s" % EMTFSiteMap[i]].fill(
                ak.flatten(self.hit_theta2[self.hit_site == i])
            )
        for i in range(0, 19):
            self.h["theta1dist_host%d" % i].fill(
                ak.flatten(self.hit_theta1[self.hit_host == i])
            )
            self.h["theta2dist_host%d" % i].fill(
                ak.flatten(self.hit_theta2[self.hit_host == i])
            )

    def FindExtraHits(self):
        sel = self.hit_segment > 1
        # print(self.hit_subsystem[sel])

    def endrun(self, outfile, nTotal=0):
        for i in range(5):
            self.h["station_map%d" % i] = self.h["station_map"][:, :, hist.loc(i)]
        super().endrun(outfile, nTotal)
