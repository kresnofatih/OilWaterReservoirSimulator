import numpy as np
# import matplotlib.pyplot as plt
from scipy.linalg import lu_solve, lu_factor
# from scipy.linalg import solve

def readpvt():
    global npvto, Rs, Pfl, Bo, Muo, Pw_ref, Bw_ref, Cw, Muw_ref, Vscw
    with open("datapvt.txt", "r") as rd:
        for i in range(0, 3):
            rd.readline()
        
        npvto = int(rd.readline())
        print(npvto)

        for i in range(0, 4):
            rd.readline()
        
        Rs = np.zeros(npvto, dtype=float)
        Pfl = np.zeros(npvto, dtype=float)
        Bo = np.zeros(npvto, dtype=float)
        Muo = np.zeros(npvto, dtype=float)
        
        for i in range(0, npvto):
            line = rd.readline()
            temp = np.array(line.split(), dtype=float)
            Rs[i] = temp[0]
            Pfl[i] = temp[1]
            Bo[i] = temp[2]
            Muo[i] = temp[3]
            print(Rs[i], Pfl[i], Bo[i], Muo[i])
        
        for i in range(0, 4):
            rd.readline()
        
        line = rd.readline()
        temp = np.array(line.split(), dtype=float)
        Pw_ref = temp[0]
        Bw_ref = temp[1]
        Cw = temp[2]
        Muw_ref = temp[3]
        Vscw = temp[4]
        print(Vscw)
    return

def readSim():
    global Ngx, Ngy, Ngz, Dx, Dy, Dz, Pi, Swi
    global phi_ref, cr, p_ref, kx, ky, kz, nrock
    global Sw, Krw, Kro, Pcow, ros, rgs, rws, Pg, Bg, Mug
    with open("datasim.txt", "r") as rr:
        for i in range(0, 5):
            rr.readline()
        
        line = rr.readline()
        temp = np.array(line.split(), dtype=int)
        Ngx = temp[0]
        Ngy = temp[1]
        Ngz = temp[2]
        print(Ngz)

        for i in range(0, 5):
            rr.readline()
        
        line = rr.readline()
        temp = np.array(line.split(), dtype=float)
        Dx = temp[0]
        Dy = temp[1]
        Dz = temp[2]
        print(Dx, Dy, Dz)

        for i in range(0, 5):
            rr.readline()
        
        line = rr.readline()
        temp = np.array(line.split(), dtype=float)
        Pi = temp[0]
        Swi = temp[1]
        print(Swi)

        for i in range(0, 6):
            rr.readline()
        
        line = rr.readline()
        temp = np.array(line.split(), dtype=float)
        phi_ref = temp[0]
        cr = temp[1]
        p_ref = temp[2]
        print(p_ref)

        for i in range(0, 5):
            rr.readline()
        
        line = rr.readline()
        temp = np.array(line.split(), dtype=float)
        kx = temp[0]
        ky = temp[1]
        kz = temp[2]
        print(kx, ky, kz)

        for i in range(0, 4):
            rr.readline()
        
        nrock = int(rr.readline())
        print(nrock)
        Sw = np.zeros(nrock, dtype=float)
        Krw = np.zeros(nrock, dtype=float)
        Kro = np.zeros(nrock, dtype=float)
        Pcow = np.zeros(nrock, dtype=float)
        # print(Sw)

        for i in range(0, 6):
            rr.readline()

        for i in range(0, nrock):
            line = rr.readline()
            temp = np.array(line.split(), dtype=float)
            Sw[i] = temp[0]
            Krw[i] = temp[1]
            Kro[i] = temp[2]
            Pcow[i] = temp[3]
            print(Sw[i], Krw[i], Kro[i], Pcow[i])
        
        for i in range(0, 5):
            rr.readline()
        
        line = rr.readline()
        temp = np.array(line.split(), dtype=float)
        ros = temp[0]
        rgs = temp[1]
        rws = temp[2]
        print(ros, rgs, rws)

        for i in range(0, 26):
            rr.readline()
        
        Pg = np.zeros(11, dtype=float)
        Bg = np.zeros(11, dtype=float)
        Mug = np.zeros(11, dtype=float)

        for i in range(0, 11):
            line = rr.readline()
            temp = np.array(line.split(), dtype=float)
            Pg[i] = temp[0]
            Bg[i] = temp[1]
            Mug[i] = temp[2]
            print(Pg[i], Bg[i], Mug[i])
        
        for i in range(0, 13):
            rr.readline()
        
        global Nw, wlx, wly, wlz, wrv, wr
        Nw = int(rr.readline()) # numOfWell
        wlx = np.zeros(Nw, dtype=int)
        wly = np.zeros(Nw, dtype=int)
        wlz = np.zeros(Nw, dtype=int)

        for i in range(0, 4):
            rr.readline()
        
        for i in range(0, Nw):
            line = rr.readline()
            temp = np.array(line.split(), dtype=int)
            wlx[i] = temp[0]-1
            wly[i] = temp[1]-1
            wlz[i] = temp[2]-1
            print(wlx[i], wly[i], wlz[i])
        
        for i in range(0, 4):
            rr.readline()
        
        wrv = np.zeros(Nw, dtype=int)
        for i in range(0, Nw):
            wrv[i] = int(rr.readline())
        
        for i in range(0, 3):
            rr.readline()
        
        wr = np.zeros((Nw, np.amax(wrv), 2), dtype=float)
        for i in range(0, Nw):
            rr.readline()
            for j in range(0, wrv[i]):
                line = rr.readline()
                temp = np.array(line.split(), dtype=float)
                wr[i][j][0] = temp[0] #time
                wr[i][j][1] = temp[1] #rate
                print(wr[i][j][0], wr[i][j][1])

        return

def interpolate(tabx, taby, x):
    i = 0
    if (x<tabx[i]):
        y = taby[i]
    elif (x>tabx[-1]):
        y = taby[-1]
    else:
        x1 = tabx[i]
        x2 = tabx[i+1]
        y1 = taby[i]
        y2 = taby[i+1]
        while(x>tabx[i+1]):
            i += 1
            x1 = tabx[i]
            x2 = tabx[i + 1]
            y1 = taby[i]
            y2 = taby[i + 1]
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
    return y

def fBo(p):
    y = interpolate(Pfl, Bo, p)
    return y
def fMuo(p):
    y = interpolate(Pfl, Muo, p)
    return y
def fRs(p):
    y = interpolate(Pfl, Rs, p)*1000/5.6146
    return y
def fBw(p):
    x = Cw*(p-Pw_ref)
    y = Bw_ref/(1+x+(x*x/2))
    return y
def fMuw(p):
    x = -Cw*(p-Pw_ref)
    y = Muw_ref/(1+x+(x*x/2))
    return y

def fdBo(p):
    y1 = fBo(p)
    p2 = p-0.001
    y2 = fBo(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx
def fdMuo(p):
    y1 = fMuo(p)
    p2 = p-0.001
    y2 = fMuo(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx
def fdBw(p):
    y1 = fBw(p)
    p2 = p-0.001
    y2 = fBw(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx
def fdMuw(p):
    y1 = fMuw(p)
    p2 = p-0.001
    y2 = fMuw(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx
def fdRs(p):
    y1 = fRs(p)
    p2 = p-0.001
    y2 = fRs(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx

def fBg(p):
    y = interpolate(Pg, Bg, p)*1000*5.6146
    return y

def fDno(p):
    y = ((ros+fRs(p)*rgs)/fBo(p))/144
    return y

def fDng(p):
    y = (rgs/fBg(p))/144
    return y

def fDnw(p):
    y = (rws/fBw(p))/144
    return y

def fphi(p):
    y = phi_ref*np.exp(cr*(p-p_ref))
    return y

def fdDno(p):
    y1 = fDno(p)
    p2 = p-0.001
    y2 = fDno(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx

def fdDng(p):
    y1 = fDng(p)
    p2 = p-0.001
    y2 = fDng(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx

def fdDnw(p):
    y1 = fDnw(p)
    p2 = p-0.001
    y2 = fDnw(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx

def fdphi(p):
    y1 = fphi(p)
    p2 = p-0.001
    y2 = fphi(p2)
    dydx = (y1-y2)/(p-p2)
    return dydx

def fkro(sw):
    y = interpolate(Sw, Kro, sw)
    return y
def fkrw(sw):
    y = interpolate(Sw, Krw, sw)
    return y
def fpcow(sw):
    y = interpolate(Sw, Pcow, sw)
    return y

def fdkro(sw):
    y1 = fkro(sw)
    sw2 = sw-0.0001
    y2 = fkro(sw2)
    dydx = (y1-y2)/(sw-sw2)
    return dydx
def fdkrw(sw):
    y1 = fkrw(sw)
    sw2 = sw-0.0001
    y2 = fkrw(sw2)
    dydx = (y1-y2)/(sw-sw2)
    return dydx
def fdpcow(sw):
    y1 = fpcow(sw)
    sw2 = sw-0.0001
    y2 = fpcow(sw2)
    dydx = (y1-y2)/(sw-sw2)
    return dydx

def initial():
    global Pg3d, Sw3d, dltx, dlty, dltz, vb, tpx, tpy, tpz, ooip, ogip, owip
    dltx = Dx/Ngx
    dlty = Dy/Ngy
    dltz = Dz/Ngz
    vb = dltx*dlty*dltz

    tpx = 6.3283*(10**(-3))*kx*dlty*dltz/dltx
    tpy = 6.3283*(10**(-3))*ky*dltx*dltz/dlty
    tpz = 6.3283*(10**(-3))*kz*dlty*dltx/dltz

    # mencari tekanan masing-masing kedalaman
    Pz = []
    Pz.append(Pi)
    # print(Pi)
    i = 0
    while i<Ngz-1:
        e = 100
        Pa = Pz[i]
        Pb = Pa
        while e>0.000001:
            pmid = 0.5*(Pa+Pb)
            dp = 0.001
            fpb = Pa+fDno(pmid)*dltz-Pb
            pmid2 = 0.5*(Pa+Pb+dp)
            fpb2 = Pa+fDno(pmid2)*dltz-(Pb+dp)
            dfpb = (fpb-fpb2)/(-dp)
            Pbb = Pb-fpb/dfpb
            e = abs(Pbb-Pb)
            Pb = Pbb
        # print(Pb)
        Pz.append(Pb)
        i+=1
    print(Pz)
    
    Pg3d = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    Sw3d = np.zeros((Ngx, Ngy, Ngz), dtype=float)

    for i in range(0, Ngx):
        for j in range(0, Ngy):
            for k in range(0, Ngz):
                Pg3d[i][j][k] = Pz[k]
                Sw3d[i][j][k] = Swi
    sumoil = 0
    sumgas = 0
    sumwat = 0

    for i in range(0, Ngx):
        for j in range(0, Ngy):
            for k in range(0, Ngz):
                sumoil += vb*fphi(Pg3d[i][j][k])*(1-Sw3d[i][j][k])/fBo(Pg3d[i][j][k])
                sumgas += vb*fRs(Pg3d[i][j][k])*fphi(Pg3d[i][j][k])*(1-Sw3d[i][j][k])/fBo(Pg3d[i][j][k])
                sumwat += vb*fphi(Pg3d[i][j][k])*Sw3d[i][j][k]/fBw(Pg3d[i][j][k])
    ooip = round(sumoil/5.6146, 3)
    ogip = round(sumgas, 3)
    owip = round(sumwat/5.6146, 3)
    print(ooip, ogip, owip)
    return

def calc_rem():
    global roip, rwip
    sumoil = 0
    sumwat = 0

    for i in range(0, Ngx):
        for j in range(0, Ngy):
            for k in range(0, Ngz):
                sumoil += vb*fphi(Pg3d[i][j][k])*(1-Sw3d[i][j][k])/fBo(Pg3d[i][j][k])
                sumwat += vb*fphi(Pg3d[i][j][k])*Sw3d[i][j][k]/fBw(Pg3d[i][j][k])
    roip = round(sumoil/5.6146, 3)
    rwip = round(sumwat/5.6146, 3)
    return

def well(time):
    global qo, qw, dQodp, dQwdp, dQods, dQwds

    # initialize
    qw= np.zeros(Nw, dtype=float)
    qo= np.zeros(Nw, dtype=float)
    dQodp = np.zeros(Nw, dtype=float)
    dQwdp = np.zeros(Nw, dtype=float)
    dQods = np.zeros(Nw, dtype=float)
    dQwds = np.zeros(Nw, dtype=float)

    for i in range(0, Nw):
        # loop over each well
        nwx = wlx[i]
        nwy = wly[i]
        nwz = wlz[i]
        pwell = Pg3d[nwx][nwy][nwz]
        swwell = Sw3d[nwx][nwy][nwz]
        # print("pwell: ", pwell)
        # print("swwell: ", swwell)

        nr = 0
        while time>wr[i][nr][0]:
            nr += 1
        
        qtot = wr[i][nr][1]*5.6146

        if i == 0:
            wc = 1  # watercut
        else:
            mobrat = (fMuo(pwell)*fBo(pwell)*fkrw(swwell))/(fMuw(pwell)*fBw(pwell)*fkro(swwell))
            wc = mobrat/(1+mobrat)
            # print("mobrat: ", mobrat)
        
        qw[i] = qtot*wc
        qo[i] = qtot*(1-wc)
        # print("fkro(swwell): ", fkro(swwell))

        dQwds[i] = qw[i]*(1-wc)*(fdkrw(swwell)/fkrw(swwell)-fdkro(swwell)/fkro(swwell))
        dQods[i] = -dQwds[i]
        dQwdp[i] = qw[i]*(1-wc)*((fdMuo(pwell)/fMuo(pwell)+fdBo(pwell)/fBo(pwell))-(fdMuw(pwell)/fMuw(pwell)+fdBw(pwell)/fBw(pwell)))
        dQodp[i] = -dQwdp[i]
    return

def poten():
    global ibw, icw, idw, iew, ifw, igw, ifo, igo

    ibw = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    icw = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    idw = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    iew = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    ifw = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    igw = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    ifo = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    igo = np.zeros((Ngx, Ngy, Ngz), dtype=float)

    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                ps = Pg3d[i][j][k]

                # ibo (Left)
                if (i>0):
                    pn = Pg3d[i-1][j][k]
                    potw = pn-ps
                    if potw > 0:
                        ibw[i][j][k]=1

                # ico (Right)
                if (i<Ngx-1):
                    pn = Pg3d[i+1][j][k]
                    potw = pn-ps
                    if potw > 0:
                        icw[i][j][k]=1
                
                # ido (Back)
                if (j>0):
                    pn = Pg3d[i][j-1][k]
                    potw = pn-ps
                    if potw > 0:
                        idw[i][j][k]=1

                # ieo (Front)
                if (j<Ngy-1):
                    pn = Pg3d[i][j+1][k]
                    potw = pn-ps
                    if potw > 0:
                        iew[i][j][k]=1
                
                # ifo (Top)
                if (k>0):
                    pn = Pg3d[i][j][k-1]
                    pm = 0.5*(pn+ps)
                    potw = pn-ps+fDnw(pm)*dltz
                    poto = pn-ps+fDno(pm)*dltz
                    if potw > 0:
                        ifw[i][j][k]=1
                    if poto > 0:
                        ifo[i][j][k]=1
                
                # igo (Top)
                if (k<Ngz-1):
                    pn = Pg3d[i][j][k+1]
                    pm = 0.5*(pn+ps)
                    potw = pn-ps-fDnw(pm)*dltz
                    poto = pn-ps-fDno(pm)*dltz
                    if potw > 0:
                        igw[i][j][k]=1
                    if poto > 0:
                        igo[i][j][k]=1
    return

def deriv(sws, swn, ps, pn, dz, ijkw, ijko, pgeo):
    global dT, fTw, fTo
    pmid = 0.5*(pn+ps)
    if ijkw == 1:
        Krwn = fkrw(swn)
        dKrwn = fdkrw(swn)
    else :
        Krwn = fkrw(sws)
        dKrwn = 0

    if ijko == 1:
        Kron = fkro(swn)
        dKron = fdkro(swn)
    else:
        Kron = fkro(sws)
        dKron = 0

    if (dz==0):
        dno = 0
        dnw = 0
        ddno = 0
        ddnw = 0
    else:
        dno = fDno(pmid)*dz
        dnw = fDnw(pmid)*dz
        ddno = fdDno(pmid)/2*dz
        ddnw = fdDnw(pmid)/2*dz
    
    Tw = Krwn/(fMuw(pmid)*fBw(pmid))*pgeo
    To = Kron/(fMuo(pmid)*fBo(pmid))*pgeo

    dT = np.zeros(4, dtype=float)
    dT[0] = dKron*pgeo/(fMuo(pmid)*fBo(pmid))*(pn-ps-dno)
    dT[1] = -To/2*(fdMuo(pmid)/fMuo(pmid)+fdBo(pmid)/fBo(pmid))*(pn-ps-dno)+To*(1-ddno)
    dT[2] = dKrwn*pgeo/(fMuw(pmid)*fBw(pmid))*(pn-ps-dnw)
    dT[3] = -Tw/2*(fdMuw(pmid)/fMuw(pmid)+fdBw(pmid)/fBw(pmid))*(pn-ps-dnw)+Tw*(1-ddnw)
    fTo = To*(pn-ps-dno)
    fTw = Tw*(pn-ps-dnw)
    return

def jacob():
    global Ja, Jb, Jc, Jd, Je, Jf, Jg, Fw, Fo
    Ja = np.zeros((Ngx, Ngy, Ngz, 4), dtype=float)
    Jb = np.zeros((Ngx+1, Ngy, Ngz, 4), dtype=float)
    Jc = np.zeros((Ngx+1, Ngy, Ngz, 4), dtype=float)
    Jd = np.zeros((Ngx, Ngy+1, Ngz, 4), dtype=float)
    Je = np.zeros((Ngx, Ngy+1, Ngz, 4), dtype=float)
    Jf = np.zeros((Ngx, Ngy, Ngz+1, 4), dtype=float)
    Jg = np.zeros((Ngx, Ngy, Ngz+1, 4), dtype=float)

    fluxo = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    fluxw = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    
    Fo = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    Fw = np.zeros((Ngx, Ngy, Ngz), dtype=float)

    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                ps = Pg3d[i][j][k]
                sws = Sw3d[i][j][k]

                # nb
                if i==0:
                    bfo = 0
                    bfw = 0
                else:
                    pn = Pg3d[i-1][j][k]
                    swn = Sw3d[i-1][j][k]
                    d = 0
                    deriv(sws, swn, ps, pn, d, ibw[i][j][k], ibw[i][j][k], tpx)
                    bfo = fTo
                    bfw = fTw
                    for l in range(0, 4):
                        Jb[i][j][k][l] = dT[l]

                # nc
                if i==Ngx-1:
                    cfo = 0
                    cfw = 0
                else:
                    pn = Pg3d[i+1][j][k]
                    swn = Sw3d[i+1][j][k]
                    d = 0
                    deriv(sws, swn, ps, pn, d, icw[i][j][k], icw[i][j][k], tpx)
                    cfo = fTo
                    cfw = fTw
                    for l in range(0, 4):
                        Jc[i][j][k][l] = dT[l]

                # nd
                if j==0:
                    dfo = 0
                    dfw = 0
                else:
                    pn = Pg3d[i][j-1][k]
                    swn = Sw3d[i][j-1][k]
                    d = 0
                    deriv(sws, swn, ps, pn, d, idw[i][j][k], idw[i][j][k], tpy)
                    dfo = fTo
                    dfw = fTw
                    for l in range(0, 4):
                        Jd[i][j][k][l] = dT[l]
                
                # ne
                if j==Ngy-1:
                    efo = 0
                    efw = 0
                else:
                    pn = Pg3d[i][j+1][k]
                    swn = Sw3d[i][j+1][k]
                    d = 0
                    deriv(sws, swn, ps, pn, d, iew[i][j][k], iew[i][j][k], tpy)
                    efo = fTo
                    efw = fTw
                    for l in range(0, 4):
                        Je[i][j][k][l] = dT[l]
                
                # nf
                if k==0:
                    ffo = 0
                    ffw = 0
                else:
                    pn = Pg3d[i][j][k-1]
                    swn = Sw3d[i][j][k-1]
                    d = -dltz
                    deriv(sws, swn, ps, pn, d, ifw[i][j][k], ifo[i][j][k], tpz)
                    ffo = fTo
                    ffw = fTw
                    for l in range(0, 4):
                        Jf[i][j][k][l] = dT[l]
                
                # ng
                if k==Ngz-1:
                    gfo = 0
                    gfw = 0
                else:
                    pn = Pg3d[i][j][k+1]
                    swn = Sw3d[i][j][k+1]
                    d = dltz
                    deriv(sws, swn, ps, pn, d, igw[i][j][k], igo[i][j][k], tpz)
                    gfo = fTo
                    gfw = fTw
                    for l in range(0, 4):
                        Jg[i][j][k][l] = dT[l]

                fluxw[i][j][k] = bfw+cfw+dfw+efw+ffw+gfw
                fluxo[i][j][k] = bfo+cfo+dfo+efo+ffo+gfo

    acc = np.zeros(4, dtype=float)

    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                ps = Pg3d[i][j][k]
                pn = P[i][j][k]     # p previous
                sws = Sw3d[i][j][k]
                swn = S[i][j][k]   # sw previous

                accw = -vb/dt*(fphi(ps)*sws/fBw(ps)-fphi(pn)*swn/fBw(pn))
                acco = -vb/dt*(fphi(ps)*(1-sws)/fBo(ps)-fphi(pn)*(1-swn)/fBo(pn))

                acc[0] = vb/dt*(fphi(ps)/fBo(ps))
                acc[1] = -vb/dt*(1-sws)*((fdphi(ps)*fBo(ps)-fphi(ps)*fdBo(ps))/(fBo(ps)*fBo(ps)))
                acc[2] = -vb/dt*(fphi(ps)/fBw(ps))
                acc[3] = -vb/dt*sws*((fdphi(ps)*fBw(ps)-fphi(ps)*fdBw(ps))/(fBw(ps)**2))

                sso = 0
                ssw = 0
                ss = np.zeros(4, dtype=float)
                for l in range(0, Nw):
                    if (i==wlx[l] and j==wly[l] and k==wlz[l]):
                        sso = qo[l]
                        ssw = qw[l]
                        ss[0] = dQods[l]
                        ss[1] = dQodp[l]
                        ss[2] = dQwds[l]
                        ss[3] = dQwdp[l]
                
                for l in range(0, 4):
                    Ja[i][j][k][l] = -Jb[i+1][j][k][l]-Jc[i-1][j][k][l]-Jd[i][j+1][k][l]
                    Ja[i][j][k][l] = Ja[i][j][k][l]-Je[i][j-1][k][l]-Jf[i][j][k+1][l]-Jg[i][j][k-1][l]
                    Ja[i][j][k][l] = Ja[i][j][k][l] + acc[l]-ss[l]
                
                Fo[i][j][k] = fluxo[i][j][k]+acco-sso
                Fw[i][j][k] = fluxw[i][j][k]+accw-ssw
    return

def jm_positioner():
    global jp

    # Absensi Jacobian
    jp = np.zeros((Ngx, Ngy, Ngz, 7), dtype=int)
    count = 0
    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                # Location A relative to i,j,k
                jp[i][j][k][0] = count
                count +=1
    # Neighbour Coordinator
    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                # Location F relative of i,j,k
                # Up
                if(k!=0):
                    jp[i][j][k][1] = jp[i][j][k-1][0]
                else:
                    jp[i][j][k][1] = -1
                # Location D relative of i,j,k
                # Back
                if(j!=0):
                    jp[i][j][k][2] = jp[i][j-1][k][0]
                else:
                    jp[i][j][k][2] = -1
                # Location B relative of i,j,k
                # Left
                if(i!=0):
                    jp[i][j][k][3] = jp[i-1][j][k][0]
                else:
                    jp[i][j][k][3] = -1
                # Location C relative of i,j,k
                # Right
                if(i!=Ngx-1):
                    jp[i][j][k][4] = jp[i+1][j][k][0]
                else:
                    jp[i][j][k][4] = -1
                # Location E relative of i,j,k
                # Front
                if(j!=Ngy-1):
                    jp[i][j][k][5] = jp[i][j+1][k][0]
                else:
                    jp[i][j][k][5] = -1
                # Location G relative of i,j,k
                # Down
                if(k!=Ngz-1):
                    jp[i][j][k][6] = jp[i][j][k+1][0]
                else:
                    jp[i][j][k][6] = -1
    return

def jm_constructor():
    global jm, jmm
    jm = np.zeros((Ngx*Ngy*Ngz*2, 2*Ngx*Ngy*Ngz), dtype=float)
    n = 0
    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                # 2-Rows per grid
                for h in range(0, 2):   # h={0,1}
                    # 7 Derivate Members
                    for m in range(0, 7):
                        # if(jp[i][j][k][m]!=-1):
                        for mm in range(0, 2):
                            if(m==0 and jp[i][j][k][m]!=-1):   # A
                                # jm[n][jp[i][j][k][m] * 2 + mm] = 1111
                                jm[n][jp[i][j][k][m] * 2 + mm] = Ja[i][j][k][h * 2+mm]
                            elif(m==1 and jp[i][j][k][m]!=-1): # F
                                # jm[n][jp[i][j][k][m] * 2 + mm] = 6666
                                jm[n][jp[i][j][k][m] * 2 + mm] = Jf[i][j][k][h * 2+mm]
                            elif(m==2 and jp[i][j][k][m]!=-1): # D
                                # jm[n][jp[i][j][k][m] * 2 + mm] = 4444
                                jm[n][jp[i][j][k][m] * 2 + mm] = Jd[i][j][k][h * 2+mm]
                            elif(m==3 and jp[i][j][k][m]!=-1): # B
                                # jm[n][jp[i][j][k][m] * 2 + mm] = 2222
                                jm[n][jp[i][j][k][m] * 2 + mm] = Jb[i][j][k][h * 2+mm]
                            elif(m==4 and jp[i][j][k][m]!=-1): # C
                                # jm[n][jp[i][j][k][m] * 2 + mm] = 3333
                                jm[n][jp[i][j][k][m] * 2 + mm] = Jc[i][j][k][h * 2+mm]
                            elif(m==5 and jp[i][j][k][m]!=-1): # E
                                # jm[n][jp[i][j][k][m] * 2 + mm] = 5555
                                jm[n][jp[i][j][k][m] * 2 + mm] = Je[i][j][k][h * 2+mm]
                            elif(m==6 and jp[i][j][k][m]!=-1):  # G
                                # jm[n][jp[i][j][k][m] * 2 + mm] = 7777
                                jm[n][jp[i][j][k][m] * 2 + mm] = Jg[i][j][k][h * 2+mm]
                    n+=1
    nrow = 0
    jmm = np.zeros(2*Ngx*Ngy*Ngz, dtype=float)
    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                jmm[nrow] = -Fo[i][j][k]
                nrow+=1
                jmm[nrow] = -Fw[i][j][k]
                nrow+=1
    # if nnn == 1:
    #     with open("jcb.txt", "w+") as ww:
    #         for r in range(0, 2*Ngx*Ngy*Ngz):
    #             for c in range(0, 2*Ngx*Ngy*Ngz):
    #                 if(c!=2*Ngx*Ngy*Ngz-1):
    #                     ww.write(str(jm[r][c])+" ")
    #                 else:
    #                     ww.write(str(jm[r][c]))
    #                     ww.write("\n")
    return

# Main Program

# Collecting Arrays
aTIME = []
aDT = []
aWATINJ = []
aOILPROD = []
aWATPROD = []
aWC = []
aWOR = []
aCUMINJ = []
aCUMOPROD = []
aCUMWPROD = []
aPWBINJ = []
aPWBPROD = []
aMB_ERR_OIL = []
aMB_ERR_WAT = []



print("Subprogram:Readdata/running")
readpvt()
readSim()
print("Subprogram:Readdata/success")
print("")

print("Subprogram:Initial_Cond/running")
initial()
print("Subprogram:Initial_Cond/success")
print("")

print("Subprogram:jm_positioner/running")
jm_positioner()
print("Subprogram:jm_positioner/success")
print("")

E_s = float(0.001)
E_p = float(0.1)
E_fo = float(1)
E_fw = float(5)

dSLIM = 0.02
dPLIM = 50

t = 0
dt = 2
tmax = 2000
cum_oilprod = 0
cum_watprod = 0
cum_watinj = 0
cum_oilinj = 0

while t<tmax:
    P = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    S = np.zeros((Ngx, Ngy, Ngz), dtype=float)
    
    t = t + dt
    for k in range(0, Ngz):
        for j in range(0, Ngy):
            for i in range(0, Ngx):
                P[i][j][k]=Pg3d[i][j][k]
                S[i][j][k]=Sw3d[i][j][k]
    
    print("Subprogram:Poten/running")
    poten()
    print("Subprogram:Poten/success")
    print("")

    c = 0
    niter = 0
    itermax = 100
    while c==0:
        niter += 1
        print("Subprogram:Well/running")
        well(t)
        print("Subprogram:Well/success")
        print("")

        print("Subprogram:Jacob/running")
        jacob()
        print("Subprogram:Jacob/success")
        print("")

        print("Subprogram:jm_creator/running")
        jm_constructor()
        print("Subprogram:jm_creator/success")
        print("")

        print("Subprogram:gauss/running")
        # sol = solve(jm, jmm)
        lu, piv = lu_factor(jm)
        sol = lu_solve((lu, piv), jmm)
        print("time: ", t)
        print("iter: ", niter)
        print("Subprogram:gauss/success")
        print("")

        # Update Values
        # Separate Solution to Sw & P
        x_dsw = np.zeros((Ngx, Ngy, Ngz), dtype=float)
        x_dp = np.zeros((Ngx, Ngy, Ngz), dtype=float)
        dr = 0
        for k in range(0, Ngz):
            for j in range(0, Ngy):
                for i in range(0, Ngx):
                    x_dsw[i][j][k] = sol[dr]
                    dr+=1
                    x_dp[i][j][k] = sol[dr]
                    dr+=1
        for k in range(0, Ngz):
            for j in range(0, Ngy):
                for i in range(0, Ngx):
                    Sw3d[i][j][k] = Sw3d[i][j][k]+x_dsw[i][j][k]
                    Pg3d[i][j][k] = Pg3d[i][j][k]+x_dp[i][j][k]
        x_dsw_max = np.amax(abs(x_dsw))
        x_dp_max = np.amax(abs(x_dp))
        fo_max = np.amax(abs(Fo))
        fw_max = np.amax(abs(Fw))

        # if(x_dp_max<E_p and x_dsw_max<E_s):
        #     c = 1
        if(fo_max<E_fo and fw_max<E_fw and x_dp_max<E_p and x_dsw_max<E_s):
            c = 1
        else:
            if(niter>itermax):
                # t = tmax
                dt = dt*0.5
                t=t-dt
            # if dt<10**-6:
            #     t = tmax
    
    print("Subprogram:Calc_Rem/running")
    calc_rem()
    print("Subprogram:Calc_Rem/success")
    print("")

    for i in range(0, Nw):
        if qw[i]>0:
            Qw = qw[i]
            Qo = qo[i]
            cum_watprod += Qw*dt
            # cum_watprod += Qw*dt/5.6146
            cum_oilprod += Qo*dt
            # cum_oilprod += Qo*dt/5.6146
        if qw[i]<0:
            Qi = abs(qw[i])
            cum_watinj += abs(qw[i])*dt
            # cum_watinj += abs(qw[i])*dt/5.6146
            cum_oilinj += abs(qo[i])*dt
            # cum_oilinj += abs(qo[i])*dt/5.6146

    mbew = (owip-rwip-cum_watprod+cum_watinj)/owip
    mbeo = (ooip-roip-cum_oilprod+cum_oilinj)/owip
    
    watcut = Qw/(Qo+Qw)
    wor = Qw/Qo

    aTIME.append(t)
    aDT.append(dt)
    aWATINJ.append(Qi)
    aOILPROD.append(Qo)
    aWATPROD.append(Qw)
    aWC.append(watcut)
    aWOR.append(wor)
    aCUMINJ.append(cum_watinj/1000)
    aCUMOPROD.append(cum_oilprod/1000)
    aCUMWPROD.append(cum_watprod/1000)
    aPWBINJ.append(Pg3d[0][0][4])
    aPWBPROD.append(Pg3d[4][4][4])
    aMB_ERR_OIL.append(mbeo)
    aMB_ERR_WAT.append(mbew)

    dPMAX = np.amax(abs(Pg3d-P))
    dSMAX = np.amax(abs(Sw3d-S))

    dtold = dt
    dT_new_p = dPLIM/dPMAX
    dT_new_s = dSLIM/dSMAX
    dt = dt*min([dT_new_s, dT_new_p])
    if(dt/dtold>2):
        dt = dtold*2
    if(dt>30):
        dt = 30
    if(t<tmax and t+dt>tmax):
        dt = tmax - t

with open("resultsim.txt", "w+") as ww:
    ww.write("TIME DT WATINJ OILPROD WATPROD WC WOR CUMINJ CUMOPROD CUMWPROD PWBINJ PWBPROD MBEO MBEW")
    ww.write("\n")
    ww.write("Days Days STB/D STB/D STB/D % STB/STB STB STB STB psia psia dec. dec.")
    ww.write("\n")
    for x in range(0, len(aTIME)):
        ww.write(str(aTIME[x])+ " ")
        ww.write(str(aDT[x])+ " ")
        ww.write(str(aWATINJ[x])+ " ")
        ww.write(str(aOILPROD[x])+ " ")
        ww.write(str(aWATPROD[x])+ " ")
        ww.write(str(aWC[x])+ " ")
        ww.write(str(aWOR[x])+ " ")
        ww.write(str(aCUMINJ[x])+ " ")
        ww.write(str(aCUMOPROD[x])+ " ")
        ww.write(str(aCUMWPROD[x])+ " ")
        ww.write(str(aPWBINJ[x])+ " ")
        ww.write(str(aPWBPROD[x])+ " ")
        ww.write(str(aMB_ERR_OIL[x])+ " ")
        ww.write(str(aMB_ERR_WAT[x])+ " ")
        ww.write("\n")

print("Simulation Run Completed")