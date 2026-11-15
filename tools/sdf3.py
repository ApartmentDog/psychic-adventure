import sys, math, random, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sps
from skimage import measure
import mesh as MS
import ext
from mesh import V, nrm

def ell(P, c, r, R=None):
    q = P - np.array(c, float)
    if R is not None: q = q @ R
    r = np.array(r, float)
    k0 = np.linalg.norm(q/r, axis=1); k1 = np.linalg.norm(q/(r*r), axis=1)
    return k0*(k0-1.0)/np.maximum(k1, 1e-9)

def cone(P, a, b, ra, rb):
    a = np.array(a, float); b = np.array(b, float); ba = b-a
    t = np.clip(((P-a) @ ba)/(ba @ ba), 0, 1); c = a+t[:, None]*ba
    return np.linalg.norm(P-c, axis=1) - (ra+(rb-ra)*t)

def smin(a, b, k):
    h = np.maximum(k-np.abs(a-b), 0)/k
    return np.minimum(a, b) - h*h*k*0.25

def smax(a, b, k): return -smin(-a, -b, k)

def segdist(P, a, b):
    a = np.array(a, float); b = np.array(b, float); ba = b-a
    t = np.clip(((P-a) @ ba)/(ba @ ba), 0, 1)
    return np.linalg.norm(P-(a+t[:, None]*ba), axis=1)


def belly(P, a, b, r1, r2, hint=(0, 0, -1)):
    """Ellipsoid whose long axis runs a->b (muscle belly). r1 = width (x-ish), r2 = depth (z-ish)."""
    a = np.array(a, float); b = np.array(b, float); u = b-a; L = np.linalg.norm(u); u = u/L
    n1 = np.cross(np.array(hint, float), u)
    if np.linalg.norm(n1) < 1e-6: n1 = np.array([1.0, 0, 0])
    n1 = n1/np.linalg.norm(n1); n2 = np.cross(u, n1)
    R = np.stack([u, n1, n2], 1)
    return ell(P, (a+b)/2, (L/2, r1, r2), R)

def seg2d(P, a, b):
    a = np.array(a[:2], float); b = np.array(b[:2], float); ba = b-a
    q = P[:, :2]-a; t = np.clip((q @ ba)/(ba @ ba), 0, 1)
    return np.linalg.norm(q-t[:, None]*ba, axis=1)

def groove(P, d, pts, depth, w, side=-1, zc=0.0, soft=2.0):
    """Press a crease into the surface along a polyline given in (x, y); side=-1 front (-z), +1 back."""
    dist = None
    for i in range(len(pts)-1):
        s_ = seg2d(P, pts[i], pts[i+1]); dist = s_ if dist is None else np.minimum(dist, s_)
    mask = np.clip(side*(P[:, 2]-zc)/soft, 0, 1)
    return d+depth*np.exp(-(dist/w)**2)*mask

def rbox(P, c, b, r, R=None):
    """Rounded box: half-size b, corner radius r (a muscle plate with flat face and soft edges)."""
    q = P - np.array(c, float)
    if R is not None: q = q @ R
    q = np.abs(q) - (np.array(b, float) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - r

def sell(P, c, r, p=3.0, R=None):
    """Superellipsoid (rounded-square muscle plate); first-order distance estimate."""
    q = P - np.array(c, float)
    if R is not None: q = q @ R
    r = np.array(r, float); u = np.abs(q)/r
    g = np.maximum((u**p).sum(1), 1e-12); f = g**(1.0/p)-1.0
    grad = g**(1.0/p-1.0)*np.linalg.norm(u**(p-1.0)/r, axis=1)
    return f/np.maximum(grad, 1e-6)

def front_z(fn, x, y):
    """z of the front surface of fn at (x, y), marching in from in front of the model."""
    zz = np.linspace(-9, 1.0, 401); Pz = np.stack([np.full_like(zz, x), np.full_like(zz, y), zz], 1)
    dv = fn(Pz); i = int(np.argmax(dv < 0))
    return float(zz[i]) if dv[i] < 0 else -2.5

def taubin(Vv, F, iters=6, lam=0.5, mu=-0.53):
    n = len(Vv)
    I = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    J = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sps.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr(); A.data[:] = 1
    deg = np.maximum(np.array(A.sum(1)).ravel(), 1)
    for _ in range(iters):
        for f in (lam, mu):
            Vv = Vv + f*((A @ Vv)/deg[:, None] - Vv)
    return Vv

def vnormals(Vv, F):
    fn = np.cross(Vv[F[:, 1]]-Vv[F[:, 0]], Vv[F[:, 2]]-Vv[F[:, 0]])
    vn = np.zeros_like(Vv)
    for k in range(3): np.add.at(vn, F[:, k], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True); ln[ln < 1e-9] = 1
    return vn/ln

def build_sdf(sp, seed):
    _sm = globals()['smin']
    smin = lambda a, b, k: _sm(a, b, k*1.35)
    rng = random.Random(seed); M = []
    h0, t, cw, ww, lt, at, al, fs, cl = (sp[k] for k in ('h0', 'torso', 'cw', 'ww', 'lt', 'at', 'al', 'fs', 'cl'))
    dl, pk, hs, nl, hd = sp['dl'], sp['pk'], sp['hs'], sp['nl'], sp['hand']
    Y = lambda o: h0+o*t
    k3 = lt**0.3; S = (1, -1)
    sx = cw*4.6+2.0; shy = Y(14.3)
    J = {}
    for s, n in ((1, 'L'), (-1, 'R')):
        sh = V(s*sx, shy, 0.2); el = sh+V(s*(0.9+1.7*lt), -10.6*al, 1.0); wr = el+V(s*1.2, -9.2*al, -3.4)
        hip = V(s*3.5*k3, h0-0.3, 0.4); kn = V(s*3.6*k3, 0.55*h0, -2.3); an = V(s*3.5*k3, 0.18*h0, 1.0)
        J[n] = dict(s=s, sh=sh, el=el, wr=wr, hip=hip, kn=kn, an=an)
    neck_top = Y(20.4); tailbase = V(0, h0+1.2, 3.0); tailtip = V(0, h0-0.1, 3.0+3.8*nl)

    mus = sp.get('mus', 1.0); mf = 0.82+0.18*mus
    df = 0.45+0.55*mus                      # crease depth factor (lanky stays soft)
    pf = 0.55+0.45*mus                      # how far muscle plates stand proud of the body
    crisp = _sm                             # un-scaled blend for sharp muscle edges
    hasabs = sp['abs']; nk = sp.get('nk', 1.0)
    RZ = lambda deg: MS.rotmat(0, 0, deg)

    def base(P):
        # pelvis + glutes
        d = ell(P, (0, h0+0.9, 0.5), (4.5*lt**0.35, 3.4, 3.3))
        for s in S:
            d = smin(d, ell(P, (s*2.6, h0+0.0, 2.5), (2.9*lt**0.3, 3.2, 2.6)), 1.0)
        # waist -> ribcage (inverted egg: narrow waist, wide chest)
        d = smin(d, ell(P, (0, Y(5.8), 0.2), (ww*4.05, 4.6*t, 3.0)), 2.6)
        d = smin(d, ell(P, (0, Y(11.8), 0.1), (cw*5.1, 5.0*t, 3.7)), 3.0)
        d = smin(d, ell(P, (0, Y(14.5), -0.1), (cw*5.3, 2.2*t, 3.3)), 2.0)
        # neck: short and thick, trapezius sloping straight into the shoulder
        d = smin(d, cone(P, (0, Y(15.8), 0.3), (0, neck_top, -0.6), 3.0*nk, 2.75*nk), 1.4)
        for s in S:
            d = smin(d, cone(P, (s*2.0*nk, Y(18.4), 0.6), (s*(sx-0.9), shy+0.9, 0.3), 1.2, 1.6), 1.3)
            d = smin(d, belly(P, (s*(cw*4.9), Y(14.0), 1.4), (s*(cw*4.0), Y(8.0), 1.4), 1.7, 1.3, hint=(1, 0, 0)), 1.2)   # lats
            d = smin(d, belly(P, (s*cw*4.8, Y(11.2), -0.9), (s*cw*3.7, Y(8.0), -0.9), 1.5, 1.7), 0.9)                       # serratus/ribs
            d = smin(d, belly(P, (s*3.3*cw, Y(14.0), 2.4), (s*2.6*cw, Y(10.0), 2.4), 2.4*cw, 1.0), 0.8)                     # scapula/rhomboid mass
            d = smin(d, belly(P, (s*4.0*ww, Y(7.4), -0.4), (s*3.5*ww, Y(2.6), -0.2), 1.5, 2.2), 0.9)                        # obliques
        return d

    # muscle plates sit on the measured front surface, so every build gets the same relief
    plates = []
    if pk >= 0.8:
        for s in S:
            c = (s*2.95*cw, Y(13.15)); hd_ = 1.0*mf
            plates.append(((c[0], c[1], front_z(base, *c)+hd_-0.95*pf), (3.05*cw*pk, 2.6*pk, hd_), 2.5, RZ(s*8), 0.3))
    else:
        for s in S:
            c = (s*2.7*cw, Y(12.7)); hd_ = 0.8
            plates.append(((c[0], c[1], front_z(base, *c)+hd_-0.3), (2.6*cw, 1.8, hd_), 2.6, RZ(s*8), 0.6))
    if hasabs:
        hw = 1.5*ww
        for yy, hh in ((9.85, 0.72), (8.05, 1.03), (6.0, 0.98)):
            for s in S:
                c = (s*(0.13+hw), Y(yy)); hd_ = 0.8*mf
                plates.append(((c[0], c[1], front_z(base, *c)+hd_-0.5*pf), (hw*1.06, hh*t*1.08, hd_), 2.5, None, 0.2))
        c = (0, Y(3.3)); hd_ = 0.75
        plates.append(((0, c[1], front_z(base, *c)+hd_-0.32*pf), (1.95*ww, 1.65*t, hd_), 2.6, None, 0.5))

    def torso(P):
        d = base(P)
        for c, b, pw, R, k in plates: d = crisp(d, sell(P, c, b, pw, R), k)
        # creases: sternum, pec/shoulder, collarbone, neck cords, hip "V", serratus, spine
        d = groove(P, d, [(0, Y(15.4)), (0, Y(10.7))], 0.45*df, 0.3)
        for s in S:
            d = groove(P, d, [(s*5.95*cw, Y(15.3)), (s*5.75*cw, Y(11.6))], 0.4*df, 0.42)
            d = groove(P, d, [(s*0.6, Y(15.75)), (s*(sx-1.8), Y(15.7))], 0.42, 0.3)
            d = groove(P, d, [(s*2.3*nk, neck_top-0.5), (s*0.5, Y(16.2))], 0.34, 0.3)         # sternocleidomastoid
            d = groove(P, d, [(s*4.3*ww, Y(3.6)), (s*2.9*ww, Y(1.6)), (s*1.0*ww, Y(-0.3))], 0.5*df, 0.45)
            if mus > 0.6:
                for k in range(3):
                    d = groove(P, d, [(s*(5.2-0.15*k)*cw, Y(10.1-1.0*k)), (s*(4.1-0.1*k)*cw, Y(8.9-1.0*k))], 0.28, 0.26)
        if hasabs:
            d = groove(P, d, [(0, Y(10.6)), (0, Y(1.6))], 0.3*df, 0.26)
        for yy in (16.6, 17.15, 17.7, 18.25, 18.8, 19.3):                                   # ribbed throat
            d = groove(P, d, [(-1.9*nk, Y(yy)), (1.9*nk, Y(yy))], 0.26, 0.18, zc=-1.3, soft=1.0)
        d = groove(P, d, [(0, Y(15.5)), (0, Y(1.2))], 0.45*df, 0.4, side=1)
        return d

    def full(P):
        T = torso(P); parts = [T]
        for n, j in J.items():
            s = j['s']; sh, el, wr, hip, kn, an = (j[k] for k in ('sh', 'el', 'wr', 'hip', 'kn', 'an'))
            ua, fa = el-sh, wr-el
            # ---- arm: big deltoid cap, biceps/triceps bellies, forearm tapering into a thick wrist (the hand is its own sculpt) ----
            a = ell(P, (s*sx, shy-0.6, 0.0), (2.9*dl, 2.75*dl, 2.85*dl))
            a = smin(a, cone(P, sh, el, 2.3*at, 1.85*at), 0.8)
            a = smin(a, belly(P, sh+ua*0.2+V(0, 0, -1.1), sh+ua*0.72+V(0, 0, -1.1), 1.9*at, 1.6*at*mf), 0.5)   # biceps
            a = smin(a, belly(P, sh+ua*0.12+V(s*0.2, 0, 1.2), sh+ua*0.78+V(s*0.2, 0, 1.2), 1.8*at, 1.7*at*mf), 0.5)   # triceps
            a = smin(a, ell(P, el, (1.95*at, 1.95*at, 1.95*at)), 0.6)
            a = smin(a, cone(P, el, wr, 1.95*at, 1.35*at), 0.6)
            a = smin(a, belly(P, el+fa*0.02+V(s*0.9, 0, -0.5), el+fa*0.5+V(s*0.55, 0, -0.4), 1.2*at, 1.15*at*mf), 0.45)   # brachioradialis
            a = smin(a, belly(P, el+fa*0.05+V(-s*0.6, 0, 0.6), el+fa*0.55+V(-s*0.35, 0, 0.4), 1.2*at, 1.3*at*mf), 0.45)  # flexors
            parts.append(smin(T, a, 0.8+1.5*np.clip(1-np.abs(P[..., 1]-Y(11.6))/2.4, 0, 1)))
            # ---- leg (the foot is its own sculpt) ----
            th = kn-hip; sn = an-kn
            l = cone(P, hip, kn, 3.1*lt, 2.25*lt)
            l = smin(l, belly(P, hip+th*0.12+V(0, 0, -1.7), hip+th*0.82+V(0, 0, -1.5), 1.85*lt, 1.45*lt*mf), 0.5)            # rectus femoris
            l = smin(l, belly(P, hip+th*0.1+V(s*1.95, 0, -0.5), hip+th*0.8+V(s*1.45, 0, -0.5), 1.35*lt, 1.6*lt*mf), 0.5)      # vastus lateralis
            l = smin(l, belly(P, hip+th*0.55+V(-s*1.2, 0, -0.9), hip+th*0.96+V(-s*0.95, 0, -0.9), 1.0*lt, 1.1*lt*mf), 0.45)    # vastus medialis
            l = smin(l, belly(P, hip+th*0.1+V(0, 0, 1.7), hip+th*0.85+V(0, 0, 1.3), 2.1*lt, 1.65*lt*mf), 0.6)                 # hamstrings
            l = smin(l, ell(P, kn, (2.0*lt, 2.1*lt, 2.0*lt)), 0.6)
            l = smin(l, ell(P, kn+V(0, 0, -1.7), (1.1*lt, 1.25*lt, 0.8*lt)), 0.4)                                           # kneecap
            l = smin(l, cone(P, kn, an, 1.95*lt, 1.15*lt), 0.7)
            l = smin(l, belly(P, kn+sn*0.06+V(s*0.75, 0, 1.2), kn+sn*0.55+V(s*0.55, 0, 0.9), 1.15*lt, 1.35*lt*mf), 0.45)      # gastrocnemius (2 heads)
            l = smin(l, belly(P, kn+sn*0.06+V(-s*0.75, 0, 1.2), kn+sn*0.55+V(-s*0.55, 0, 0.9), 1.15*lt, 1.35*lt*mf), 0.45)
            l = smin(l, belly(P, kn+sn*0.12+V(s*0.5, 0, -1.2), kn+sn*0.62+V(s*0.35, 0, -1.0), 0.8*lt, 0.7*lt), 0.45)           # tibialis
            parts.append(smin(T, l, 1.6))
        parts.append(smin(T, cone(P, tailbase, tailtip, 1.9, 0.85), 1.1))
        d = parts[0]
        for q in parts[1:]: d = np.minimum(d, q)
        return d

    # ---- grid + marching cubes ----
    step = float(__import__("os").environ.get("SDF_STEP", "0.27"))
    xmax = sx+6.0
    xs = np.arange(-xmax, xmax+1e-6, step); ys = np.arange(-0.4, neck_top+1.0, step); zs = np.arange(min(-8.5, -(5.5*fs+3.0)), max(9.0, tailtip[2]+2.0), step)
    X, Yg, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    P = np.stack([X.ravel(), Yg.ravel(), Z.ravel()], 1)
    vol = full(P).reshape(X.shape)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(step, step, step), gradient_direction='ascent')
    verts = verts + np.array([xs[0], ys[0], zs[0]])
    verts = taubin(verts, faces, iters=8)
    verts, faces = ext.decimate(verts, faces, int(os.environ.get('TARGET_BODY', '0')))
    # orientation sanity (outward normals)
    fn = np.cross(verts[faces[:, 1]]-verts[faces[:, 0]], verts[faces[:, 2]]-verts[faces[:, 0]])
    cen = verts[faces].mean(1)
    if np.mean(np.einsum('ij,ij->i', fn, cen-np.array([0, Y(10), 0]))) < 0: faces = faces[:, ::-1]
    # baked ambient occlusion + skylight -> ramp level per vertex
    vn = vnormals(verts, faces); occ = np.zeros(len(verts))
    for d_, w in ((0.5, 1.0), (1.0, 0.6), (1.8, 0.4)):
        sd = full(verts+vn*d_)
        occ += w*np.clip((d_-np.maximum(sd, 0))/d_, 0, 1)
    ao = np.clip(1-1.0*occ/2.0, 0, 1)
    shade_v = 0.44+0.44*ao+0.14*(0.5+0.5*vn[:, 1])+0.06*np.clip(vn @ nrm(V(-0.4, 0.5, -0.76)), 0, 1)
    lv_face = np.clip(np.round(((shade_v[faces].mean(1))-0.58)/0.033), 0, 15).astype(int)
    vlv_all = np.clip((shade_v-0.58)/0.033, 0, 15)
    # region split
    tsd = torso(verts[faces].mean(1)); fc = verts[faces].mean(1)
    cand = []
    for n, j in J.items():
        sfx = '_'+n
        cand += [('arm'+sfx, j['sh'], j['el'], 2.0*at), ('forearm'+sfx, j['el'], j['wr'], 1.6*at),
                 ('hand'+sfx, j['wr'], j['wr']+V(0, -1.5, -0.2), 1.0*at),
                 ('leg'+sfx, j['hip'], j['kn'], 2.7*lt), ('shin'+sfx, j['kn'], j['an'], 1.6*lt),
                 ('foot'+sfx, j['an'], j['an']+V(0, -1.0, 0), 0.9*lt)]
    cand.append(('tail', tailbase, tailtip, 1.4))
    dists = np.stack([segdist(fc, a, b)-r_ for _, a, b, r_ in cand], 1)
    owner = np.argmin(dists, 1)
    names = np.array([c[0] for c in cand])[owner]
    names = np.where(tsd < 0.6, 'body', names)
    for gname in sorted(set(names.tolist())):
        sel = np.where(names == gname)[0]
        used = np.unique(faces[sel]); remap = -np.ones(len(verts), int); remap[used] = np.arange(len(used))
        m = MS.Mesh('skin_'+gname, gname); m.verts = [verts[i].copy() for i in used]
        m.faces = [([int(remap[a]), int(remap[b]), int(remap[c])], 'skinramp') for a, b, c in faces[sel]]
        m.flv = [int(lv_face[i]) for i in sel]; m.vlv = [float(vlv_all[i]) for i in used]
        M.append(m)

    # ---- hands and feet: separate fine sculpts, with claws and pads as their own meshes ----
    hq = ext.QUAL == 'hi'                    # the Blockbench build sculpts hands/feet finely enough to hold skin creases
    hstep = 0.1 if hq else float(np.clip(step*0.67, 0.2, 0.45)); fstep = 0.12 if hq else float(np.clip(step*0.72, 0.22, 0.5)); ce = cl**0.5
    for n, j in J.items():
        s = j['s']; gh, gfoot, gf = f'hand_{n}', f'foot_{n}', f'forearm_{n}'
        wr, el, an, kn = j['wr'], j['el'], j['an'], j['kn']
        field, crs, claws, pads, lo, hi = ext.make_hand(wr, el, s, hd, at, ce, creases=hq)
        M.append(ext.to_mesh(f'hand_sculpt_{n}', gh, *ext.sculpt(field, lo, hi, hstep, target=int(os.environ.get('TARGET_HAND', '0')), crease=crs)))
        for i, c in enumerate(claws): M.append(ext.claw(f'claw_{n}{i}', gh, c))
        for i, pp in enumerate(pads): M.append(ext.pad_mesh(f'palmpad_{n}{i}', gh, *pp))
        field, crs, claws, lo, hi = ext.make_foot(an, kn, s, fs, ce, lt, k3, creases=hq)
        M.append(ext.to_mesh(f'foot_sculpt_{n}', gfoot, *ext.sculpt(field, lo, hi, fstep, target=int(os.environ.get('TARGET_FOOT', '0')), crease=crs)))
        for i, c in enumerate(claws): M.append(ext.claw(f'talon_{n}{i}', gfoot, c))
        furs = sp['fur']; ax = an[0]
        if 'forearm' in furs:
            MS.fur_patch(M, rng, gf, f'furarm_{n}', el+(wr-el)*0.3+V(s*2.2, 0, 0.3), V(s, 0.1, 0.2), 1.3, 5, 1.6)
        if 'heel' in furs:
            MS.fur_patch(M, rng, gfoot, f'furheel_{n}', V(ax, 3.3, an[2]+2.7), V(0, 0.25, 1), 1.0, 4, 1.7)
    furs = sp['fur']
    if 'back' in furs: MS.fur_patch(M, rng, 'body', 'furback', V(0, Y(2.2), 3.6), V(0, 0.45, 1), 2.2*ww, 8, 2.4)
    if 'mane' in furs: MS.fur_patch(M, rng, 'body', 'furmane', V(0, Y(17.0), 2.6), V(0, 0.5, 1), 3.2*cw, 10, 2.6)
    if 'chest' in furs: MS.fur_patch(M, rng, 'body', 'furchest', V(0, Y(12.5), -4.4), V(0, 0.2, -1), 2.2*cw, 7, 1.3)
    if 'tailtip' in furs: MS.fur_patch(M, rng, 'tail', 'furtail', tailtip, V(0, 0.1, 1), 0.9, 5, 2.0)
    if sp.get('horn'):
        hb0 = V(0, h0+1.9, 3.3)
        M.append(MS.spike('sacralhorn', 'tail', hb0, hb0+V(0, 2.6, 3.4), 1.1, 'claw', curve=V(0, 1.0, -0.6), N=6))

    hb = neck_top-1.0; hz = -0.9
    head = dict(x0=-hs/2, x1=hs/2, y0=hb, y1=hb+hs, z0=hz-hs/2, z1=hz+hs/2)
    groups = {'body': ((0, h0, 0), None), 'head': ((0, neck_top, hz), 'body'), 'tail': (tuple(tailbase), 'body')}
    for n, j in J.items():
        groups[f'arm_{n}'] = (tuple(j['sh']), 'body'); groups[f'forearm_{n}'] = (tuple(j['el']), f'arm_{n}')
        groups[f'hand_{n}'] = (tuple(j['wr']), f'forearm_{n}')
        groups[f'leg_{n}'] = (tuple(j['hip']), None); groups[f'shin_{n}'] = (tuple(j['kn']), f'leg_{n}')
        groups[f'foot_{n}'] = (tuple(j['an']), f'shin_{n}')
    return M, head, groups

VARS = {}
for k, v in MS.VARIANTS.items():
    nv = dict(v); nv['hipspike'] = False; nv['elbowspike'] = False
    VARS[k.replace('_smooth', '_sculpt')] = nv
# tune builds for the blended body
VARS['enderling_main_sculpt'].update(sk='#494656', clawcol=(46, 43, 62), h0=28, torso=1.0, cw=0.92, ww=0.82, lt=0.9, at=0.85, al=1.15, mus=0.75, pk=1.0, hs=9.0, fs=1.1, hand=1.2, cl=1.0, fur=['back', 'tailtip', 'heel'])
VARS['enderling_bruiser_sculpt'].update(h0=27, torso=1.0, cw=1.02, ww=0.9, lt=1.0, at=1.0, al=1.05, dl=1.1, pk=1.15, mus=1.0, hs=9.5, fs=1.25, hand=1.4, horn=True, fur=['mane', 'back', 'heel', 'tailtip'])
VARS['enderling_lanky_sculpt'].update(nk=0.82, h0=31, torso=1.12, cw=0.8, ww=0.72, lt=0.7, at=0.66, al=1.2, mus=0.4, hs=8.5, fs=1.1, hand=1.15)
VARS['enderling_ember_sculpt'].update(h0=28, torso=1.0, cw=0.92, ww=0.84, lt=0.92, at=0.86, al=1.12, mus=0.8, pk=1.0, hs=9.0, fs=1.1, hand=1.25, horn=True, fur=['back', 'tailtip', 'heel'])

def build_all(outdir=None):
    res = {}
    for i, (name, sp) in enumerate(VARS.items()):
        if os.environ.get('ONLY') and os.environ['ONLY'] not in name: continue
        rng = random.Random(700+i)
        M, head, groups = build_sdf(sp, 700+i)
        atlas, cells = MS.make_atlas(sp, rng)
        data = MS.finalize(name, sp, M, head, groups, atlas, cells, rng)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, name+'.bbmodel'), 'w') as f: json.dump(data, f, separators=(',', ':'))
        res[name] = (sp, M, head, atlas, cells, data)
        print(name, 'meshes', len(M), 'faces', sum(len(m.faces) for m in M))
    return res

if __name__ == '__main__':
    build_all(sys.argv[1])
