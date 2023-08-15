import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import math

osc_param = {
    "ordering" : "NO",
    "s2_12" : 0.303,
    "dm2_21" : 7.41e-5,
    "s2_13_N" : 0.02203,
    "dm2_31_N" : 2.511e-3,
    "s2_13_I" : 0.02219,
    "dm2_32_I" : -2.498e-3,
    "source" : "NuFIT 5.2 (2022)"
}

rho_matter = 2.45

flux_file = "/home/bjelmini/JUNO_u/Analysis/ORSA/INSS2023/ReactorAntineutrinoFluxShape.txt"


def sin2(x_, dm2_):
    appo = 1.267 * dm2_ * x_  # x in [m/MeV]
    return np.power(np.sin(appo), 2)

def sin(x_, dm2_):
    appo = 1.267 * dm2_ * x_  # x in [m/MeV]
    return np.sin(appo)

def cos(x_, dm2_):
    appo = 1.267 * dm2_ * x_  # x in [m/MeV]
    return np.cos(appo)

def potential(density, nu_energy):
    y_e = 0.5
    return -1.52e-4 * y_e * density * nu_energy * 1.e-3

def eval_vacuum_prob(x_, baseline = 52.5, nu_energy_=True, ordering="NO"):

    if nu_energy_:
        x = baseline * 1000 / x_  # [m/MeV]
    else:
        x = x_  # [m/MeV]

    s2_12 = osc_param["s2_12"]
    dm2_21 = osc_param["dm2_21"]

    if ordering == "NO":
        s2_13 = osc_param["s2_13_N"]
        dm2_31 = osc_param["dm2_31_N"]
    elif ordering == "IO":
        s2_13 = osc_param["s2_13_I"]
        dm2_31 = osc_param["dm2_32_I"] + dm2_21
    else:
        raise ValueError("Error: NO or IO!")


    aa = np.power(1 - s2_13, 2) * 4. * s2_12 * (1 - s2_12)
    bb = (1 - s2_12) * 4 * s2_13 * (1 - s2_13)
    cc = s2_12 * 4 * s2_13 * (1 - s2_13)

    vacuum_prob = (1. - aa * sin2(x, dm2_21)
                   - bb * sin2(x, dm2_31)
                   - cc * sin2(x, dm2_31 - dm2_21))
    
    return vacuum_prob

def eval_matter_prob(x_, baseline = 52.5, nu_energy_=True, ordering="NO"):

    if nu_energy_:
        x = baseline * 1000 / x_  # [m/MeV]
        nu_en = x_
    else:
        x = x_  # [m/MeV]
        nu_en = baseline * 1000 / x_

    s2_12 = osc_param["s2_12"]
    dm2_21 = osc_param["dm2_21"]

    if ordering == "NO":
        s2_13 = osc_param["s2_13_N"]
        dm2_31 = osc_param["dm2_31_N"]
    elif ordering == "IO":
        s2_13 = osc_param["s2_13_I"]
        dm2_31 = osc_param["dm2_32_I"] + dm2_21
    else:
        raise ValueError("Error: NO or IO!")

    deltam_ee = dm2_31 - s2_12 * dm2_21
    dm2_32 = dm2_31 - dm2_21
    c2_12 = 1. - s2_12
    c2_13 = 1. - s2_13
    c_2_12 = 1. - 2. * s2_12

    pot = potential(rho_matter, nu_en)
    appo_12 = c2_13 * pot / dm2_21

    s2_12_m = s2_12 * \
        (1. + 2. * c2_12 * appo_12 + 3. * c2_12 * c_2_12 * appo_12 * appo_12)
    dm2_21_m = dm2_21 * \
        (1. - c_2_12 * appo_12 + 2. * s2_12 * c2_12 * appo_12 * appo_12)
    s2_13_m = s2_13 * (1. + 2. * c2_13 * pot / deltam_ee)
    dm2_31_m = (dm2_31 \
                * (1. - pot / dm2_31 * (c2_12 * c2_13 - s2_13 \
                                             - s2_12 * c2_12 * c2_13 * appo_12)))
    dm2_32_m = (dm2_32
                * (1. - pot / dm2_32 * (s2_12 * c2_13 - s2_13 \
                                        + s2_12 * c2_12 * c2_13 * appo_12)))
    
    aa = np.power(1. - s2_13_m, 2) * 4. * s2_12_m * (1. - s2_12_m)
    bb = (1. - s2_12_m) * 4 * s2_13_m * (1. - s2_13_m)
    cc = s2_12_m * 4 * s2_13_m * (1. - s2_13_m)

    matter_prob = (1. - aa * sin2(x, dm2_21_m)
                   - bb * sin2(x, dm2_31_m)
                   - cc * sin2(x, dm2_32_m))
    
    return matter_prob

def ibd_xsection(nu_energy):

    m_n = 939.57
    m_p = 938.28
    m_e = 0.511
    delta = m_n - m_p

    positron_energy = nu_energy - delta
    positron_momentum = np.sqrt(np.power(positron_energy,2) - m_e**2)

    const = 0.96e-43

    xsec = const * positron_momentum * positron_energy

    return xsec

def read_flux(filename=flux_file):

    data = pd.read_csv(filename, header = 0, sep=",", names=["flux"])
    xx = data.index.to_numpy()
    yy = data["flux"].to_numpy()

    return xx, yy

def reactor_flux(nu_energy):

    xx, yy = read_flux()

    f_temp = interp1d(xx, yy, kind="quadratic", fill_value="extrapolate")

    return f_temp(nu_energy)

def eval_unosc_spectrum(nu_energy, baseline=180, years=10):

    power = 70  # [GW]
    nu_emission = 6e20  # [nu/GW]
    det_efficiency = 0.9 
    runtime = years * 365.25 * 24 * 60 * 60  # [s]
    nb_of_p = 5.98e31

    xsec = ibd_xsection(nu_energy) * nb_of_p  # [cm2]
    flux = reactor_flux(nu_energy)  # [nu/MeV/s]
    flux_reduction = 4 * math.pi * (baseline*1000*100)**2

    temp_sp = det_efficiency * xsec / flux_reduction * power * nu_emission * runtime * flux

    return temp_sp
