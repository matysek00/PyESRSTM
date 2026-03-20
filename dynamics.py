
import numpy as np
from scipy.optimize import curve_fit

from PyTimeESR import Electrode, QD, QME, QME_matrix, rates, current, sum_rates
from PyTimeESR.dynamics import Bexch, Trel, Szacc
from PyTimeESR.ESR import ESR_fit_fun, guess_p0

from PyTimeESR.units import Hartree, GHz

import time 

def calcV(dot, G, GC, freq):
    NFT = G.shape[-1]
    I = np.zeros((len(freq), NFT), dtype=complex)
    M = QME_matrix(G, dot.Delta, )
    for i, f in enumerate(freq):
        rho = QME(M.copy(), f, qme_matrix=True)
        I[i] = current(rho,GC)
    return I, rho


def main():
    Nint = 1e4
    NF = 1
    outfile = 'Vdc_scan.dat'
    fmt = '{:.10f} '*11 + '\n'
    p0 = None
    
    Bmag = 0.6082762560144831
    
    Vdcs = np.linspace(-40, -20, 50)
    thetas = np.linspace(0.05, np.pi, 100)

    for theta in [thetas[20]]:
        B = Bmag*np.array([[np.sin(theta), 0, np.cos(theta)]])
        dot = QD(-30, 100, np.array([0.5]), B, np.array([[2.0, 2.0, 2.0]]))
        dot_norm = QD(-30, 100, np.array([0.5]),10*B, np.array([[2.0, 2.0, 2.0]]))

        Right = Electrode(0, 0, 0., 5e-3, 1e-1, 4.5, Nint=Nint, Cutoff=300, segments=[[dot.eps,30, Nint],[dot.U+dot.eps,30, Nint]])
        GRplus, GRminus = rates(dot, Right, 1e-5, 0)
        GRplus_norm, GRminus_norm = rates(dot_norm, Right, 1e-5, 0)

        GR = GRplus + GRminus
        GR_norm = GRplus_norm + GRminus_norm
        del GRplus, GRminus, GRminus_norm, GRplus_norm

        Zeeman = dot.Delta[0,1]*Hartree/GHz
        print(Zeeman)
        freq = np.linspace(Zeeman-1, Zeeman+1, 500)
        np.save(f'freq.npy', freq)
        for Vdc in Vdcs:
            ts = time.time()
            Left = Electrode(Vdc, 5, 0.5, 1.25e-3, 1e-1, 4.5, Nint=Nint, Cutoff=300,  
                    segments=[[dot.eps,30, Nint],[dot.U+dot.eps,30,Nint] ])
            GLminus, GLplus = rates(dot, Left, 20, NF)
            GL = GLplus+GLminus
            GC = GLplus - GLminus
            G = sum_rates(GL, GR)
            I, rho = calcV(dot, G, GC, freq)
            
            Bxc0 = Bexch(GL, GR, dot.theta)*Hartree/GHz
            Bxc1 = Bexch(GL, GR, dot.theta, n=1)*Hartree/GHz
            T1 = Trel(GL, GR)*Hartree/GHz
            Sz0 = Szacc(GL, GR, rho, Left.Spin_polarization, dot.theta)*Hartree/GHz
            Sz1 = Szacc(GL, GR, rho, Left.Spin_polarization, dot.theta, n=1)*Hartree/GHz

            GLminus, GLplus = rates(dot_norm, Left, 20, NF)
            GL = GLplus+GLminus
            GC = GLplus - GLminus
            G = sum_rates(GL, GR_norm)
            I_norm, _ = calcV(dot_norm, G, GC, freq)

            esr = np.real((I-I_norm)[:,NF])
            if p0 is None:
                p0 = guess_p0(freq, esr, Zeeman)
            try: 
                popt, _ = curve_fit(ESR_fit_fun, freq, esr, p0=p0, maxfev=int(1e4))
                #np.save(f'failed-theta={theta:.3f}V={Vdc:.2f}.npy', esr)
                p0 = popt # We use the results as an estimate for the next guess
            except:
                print(f'Fitting Falied for VDC={Vdc:.1f}')
                popt = np.zeros_like(p0)
                np.save(f'failed-theta={theta:.3f}V={Vdc:.2f}.npy', esr)
                
            Lamb, gamma, Is, Ia = popt[:4]
            with open(outfile, 'a') as f:
                f.write(fmt.format(theta, Vdc, Lamb, gamma, Is, Ia, Bxc0, Bxc1, Sz0, Sz1, T1))
            tt = time.time() - ts

            print(f'Vdc={Vdc:.2f} mV done in {tt:.2} s')

if __name__ == '__main__':
    main()