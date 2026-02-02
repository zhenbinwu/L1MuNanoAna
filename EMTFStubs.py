"""
File        : EMTFStubs.py
Author      : Ben Wu
Contact     : benwu@fnal.gov
Date        : 2025 May 16

Description : Analyzer for EMTF stubs from EMTF producer
"""

import uproot
import numpy as np
import awkward as ak
from hist import Hist
import hist
from Common import Module

# EMTF stub layer mapping
emtf_layers = {
    0: "ME1/1",
    1: "ME1/2",
    2: "ME1/3",
    3: "ME2/1",
    4: "ME2/2",
    5: "ME3/1",
    6: "ME3/2",
    7: "ME4/1",
    8: "ME4/2",
    9: "GE1/1",
    10: "GE2/1",
    11: "ME0"
}

class EMTFStubs(Module):
    def __init__(self, name="EMTFStubs"):
        super().__init__(name)
        self.__bookHistograms__()

    def __GetEvent__(self, event):
        super().__GetEvent__(event)
        for k in dir(event):
            if k.startswith("emtfstub_"):
                setattr(self, k.replace("emtfstub_", ""), event[k])

    def run(self, event):
        super().run(event)
        self.__analyzeStubs__()

    def __bookHistograms__(self):
        """Book all histograms for EMTF stub analysis"""
        # Basic stub properties
        self.h.update({
            "stub_layer": Hist.new.Reg(12, 0, 12, name="Layer").Double(),
            "stub_bx": Hist.new.Reg(5, -2, 3, name="BX").Double(),
            "stub_quality": Hist.new.Reg(16, 0, 16, name="Quality").Double(),
            "stub_phi": Hist.new.Reg(100, -3.2, 3.2, name="Phi").Double(),
            "stub_eta": Hist.new.Reg(100, -2.5, 2.5, name="Eta").Double(),
            "stub_z0": Hist.new.Reg(100, -500, 500, name="Z0").Double(),
            "stub_d0": Hist.new.Reg(100, -50, 50, name="D0").Double(),
        })

        # Per-layer histograms
        for layer in range(12):
            self.h[f"stub_phi_layer{layer}"] = Hist.new.Reg(100, -3.2, 3.2, name="Phi").Double()
            self.h[f"stub_eta_layer{layer}"] = Hist.new.Reg(100, -2.5, 2.5, name="Eta").Double()

        # Track matching
        self.h.update({
            "track_stub_match": Hist.new.Reg(10, 0, 10, name="N Stubs").Double(),
            "track_stub_match_quality": Hist.new.Reg(10, 0, 10, name="Quality").Double(),
            "track_stub_match_bx": Hist.new.Reg(5, -2, 3, name="BX").Double(),
        })

        # Stub efficiency
        self.h.update({
            "stub_eff_layer": Hist.new.Reg(12, 0, 12, name="Layer").Double(),
            "stub_eff_eta": Hist.new.Reg(100, -2.5, 2.5, name="Eta").Double(),
            "stub_eff_phi": Hist.new.Reg(100, -3.2, 3.2, name="Phi").Double(),
        })

    def __analyzeStubs__(self):
        """Analyze EMTF stub properties and track matching"""
        # Fill basic stub histograms
        for stub in self.stubs:
            layer = stub.layer
            bx = stub.bx
            quality = stub.quality
            phi = stub.phi
            eta = stub.eta
            z0 = stub.z0
            d0 = stub.d0

            self.h["stub_layer"].fill(layer)
            self.h["stub_bx"].fill(bx)
            self.h["stub_quality"].fill(quality)
            self.h["stub_phi"].fill(phi)
            self.h["stub_eta"].fill(eta)
            self.h["stub_z0"].fill(z0)
            self.h["stub_d0"].fill(d0)

            # Fill per-layer histograms
            if layer < 12:
                self.h[f"stub_phi_layer{layer}"].fill(phi)
                self.h[f"stub_eta_layer{layer}"].fill(eta)

        # Analyze track-stub matching
        for track in self.tracks:
            n_stubs = len(track.stubs)
            avg_quality = np.mean([s.quality for s in track.stubs])
            avg_bx = np.mean([s.bx for s in track.stubs])

            self.h["track_stub_match"].fill(n_stubs)
            self.h["track_stub_match_quality"].fill(avg_quality)
            self.h["track_stub_match_bx"].fill(avg_bx)

    def endrun(self, outfile, nTotal=0):
        """End of run - save histograms"""
        super().endrun(outfile, nTotal)
        
        # Save histograms to file
        for hname, hist in self.h.items():
            hist.write(outfile, hname)
