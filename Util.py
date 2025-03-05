#!/usr/bin/env python
# encoding: utf-8

# File        : Util.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2024 Apr 10
#
# Description : 

import math
import numpy as np
import awkward as ak

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function for Awkward ~~~~~

def akwhere(condition, iftrue, iffalse):
    flatCond = ak.flatten(condition)
    flattrue = ak.flatten(iftrue)
    flatflase = ak.flatten(iffalse)
    flatout = ak.where(flatCond, flattrue, flatflase)
    out=ak.unflatten(flatout, ak.num(condition, axis=1))
    return out

def GetUnique(array):
    length = ak.run_lengths(array)
    flat = ak.flatten(array)
    # samearray = ak.unflatten(array, length)
    samearray = ak.unflatten(flat, ak.flatten(length))
    unique = ak.firsts(samearray)
    cnts = ak.num(length)
    return ak.unflatten(unique, cnts)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  EMTF conversion ~~~~~
def calc_phi_loc_deg_from_int(phi_int):
    loc = phi_int / 60. - 22.
    return loc

def calc_phi_loc_rad_from_int(phi_int):
    loc = np.deg2rad(calc_phi_loc_deg_from_int(phi_int))
    return loc

def calc_phi_glob_rad_from_loc(loc, sector):
    # loc in rad, sector [1..6]
    glob = np.deg2rad(calc_phi_glob_deg_from_loc(np.rad2deg(loc), sector))
    return glob

def calc_phi_glob_deg_from_loc(loc, sector):
    # loc in deg, sector [1..6]
    glob = loc + 15. + (60. * (sector - 1))
    glob = ak.where(glob >=180, glob-360, glob)
    return glob

def calc_eta_from_theta_rad(theta_rad):
    eta_calc = -1. * np.log(np.tan(theta_rad / 2.))
    eta = np.where(theta_rad == 0, np.ones_like(theta_rad) * np.inf, eta_calc)
    eta = np.where((theta_rad == np.pi) | (theta_rad == -np.pi), -1. * np.ones_like(theta_rad) * np.inf, eta)
    return eta

def calc_theta_rad_from_int(theta_int):
    theta = np.deg2rad(calc_theta_deg_from_int(theta_int))
    return theta

def calc_theta_deg_from_int(theta_int):
    theta = theta_int * (45.0 - 8.5) / 128. + 8.5
    return theta

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Propogation of Displaced GenMuon ~~~~~

## KMTF propogation
def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = p1 - p2
    res = ak.where (res > math.pi, res-2*math.pi, res)
    res = ak.where (res < -1 *math.pi, res+2*math.pi, res)
    return res

def akgetSt2Eta(lxy, vz, eta, st2_r=512):
    ## From getSt2Eta
    theta_mu = 2*np.arctan(np.exp(-1 *eta))
    st2_z = (st2_r-lxy)/np.tan(theta_mu)+vz
    theta_st2 = np.arctan2(st2_r,st2_z)
    eta_st2 = -1*np.log(np.tan(theta_st2/2.))
    return eta_st2

def akgetSt2Phi(vx, vy, genphi, r=512):  
    x1 = vx
    y1 = vy
    x2 = vx + np.cos(genphi)
    y2 = vy + np.sin(genphi)
    dx = x2-x1
    dy = y2-y1
    dr = np.sqrt(dx**2+dy**2)
    D = x1*y2-x2*y1
    delta = (r**2)*(dr**2)-D**2

    retphi  = ak.zeros_like(x1)
    retphi = ak.where(delta < 0, np.arctan2(y1, x1), retphi)

    xP = (D*dy+np.copysign(1,dy)*dx*np.sqrt(delta))/dr**2
    xM = (D*dy-np.copysign(1,dy)*dx*np.sqrt(delta))/dr**2
    yP = (-D*dx+abs(dy)*np.sqrt(delta))/dr**2
    yM = (-D*dx-abs(dy)*np.sqrt(delta))/dr**2
    
    phi1 = np.arctan2(yP,xP)
    phi2 = np.arctan2(yM,xM)
    
    dphi1 = abs(deltaPhi(phi1,  genphi))
    dphi2 = abs(deltaPhi(phi2,  genphi))

    retphi2 = ak.where(dphi1 < dphi2, phi1, phi2)
    retphi = ak.where(delta < 0 , retphi, retphi2)

    return retphi

def getKMTFEta1(lxy, vz):
    theta1 = np.arctan((700.-lxy)/(650.-vz))
    theta1 = ak.where(theta1 < 0, math.pi+theta1, theta1)
    eta1 = -np.log(np.tan(theta1/2.0))
    return eta1

def getKMTFEta2(lxy, vz):
    theta2 = math.pi-np.arctan((700.-lxy)/(650.+vz))
    theta2 = ak.where( theta2 > math.pi, theta2-math.pi, theta2)
    eta2 = -np.log(np.tan(theta2/2.0))
    return eta2                                     

def getKMTFAcceptance(lxy, vz, eta):
    eta1 = getKMTFEta1(lxy, vz)
    eta2 = getKMTFEta2(lxy, vz)
    return (eta < eta1 ) & (eta > eta2)

## EMTF propogation
def calc_etaphi_star_simple(vx, vy, vz, eta, phi, me2_z=808.0):
    me2_dz =ak.where(eta>0, abs(me2_z - vz) / abs(np.sinh(eta)), abs(-me2_z - vz) / abs(np.sinh(eta)))

    x_star = vx + me2_dz * np.cos(phi)
    y_star = vy + me2_dz * np.sin(phi)
    r_star = np.sqrt(x_star ** 2 + y_star ** 2)
    eta_star = np.arcsinh(me2_z / r_star) * (eta / abs(eta))

    phi_star = np.arctan(y_star/x_star)
    phi_star = ak.where((x_star < 0) & (y_star >=0), math.pi + phi_star, phi_star  ) 
    phi_star = ak.where((x_star < 0) & (y_star <=0), phi_star - math.pi, phi_star  ) 

    return eta_star, phi_star
