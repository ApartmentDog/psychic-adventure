"""Hands and feet as their own fine-resolution sculpts (separate SDF volumes), plus loft claws.

Each make_* returns (field, claws, lo, hi):
  field(P) -> (skin distance, pad field)   pad field < ~0 marks palm-pad surface
  claws    -> list of (c0, c1, c2, half-width, half-height, width-axis) for claw()
"""
import math
import numpy as np
import scipy.sparse as sps
from skimage import measure
import mesh as MS
from mesh import V, nrm


def sm(a, b, k):
    h = np.maximum(k-np.abs(a-b), 0)/k
    return np.minimum(a, b) - h*h*k*0.25


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


def sph(P, c, r): return np.linalg.norm(P-np.array(c, float), axis=1)-r


def evalprims(P, pr):
    d = None
    for p in pr:
        if p[0] == 'cone': e = cone(P, p[1], p[2], p[3], p[4])
        elif p[0] == 'ell': e = ell(P, p[1], p[2], p[3])
        else: e = sph(P, p[1], p[2])
        k = p[-1]
        d = e if d is None else (sm(d, e, k) if k else np.minimum(d, e))
    return d


QUAL = __import__('os').environ.get('QUAL', 'hi')          # hi (Blockbench) | near | far (in-game levels)
CLAW_RES = {'hi': (10, 8), 'near': (8, 6), 'far': (5, 4)}[QUAL]
PAD_RES = {'hi': (8, 4), 'near': (6, 3), 'far': (3, 1)}[QUAL]


def claw(name, group, spec):
    """Claw as drawn: a broad curved blade with a ridge down its back and a flatter underside, hooking to a point.
    spec = (c0, c1, c2, half-width, ridge height, width axis, 'up' = the back of the digit)."""
    c0, c1, c2, rw, rh, ref, up = spec; sides, nr = CLAW_RES
    ts = np.linspace(0, 1, nr); knots = [0, 0.25, 0.5, 0.72, 0.9, 1.0]
    wprof = np.interp(ts, knots, [1.0, 1.0, 0.94, 0.76, 0.45, 0.05])
    hprof = np.interp(ts, knots, [1.0, 0.96, 0.86, 0.68, 0.4, 0.05])
    m = MS.Mesh(name, group); rings = []
    for t, fw, fh in zip(ts, wprof, hprof):
        p = (1-t)**2*c0 + 2*t*(1-t)*c1 + t*t*c2; tan = 2*(1-t)*(c1-c0) + 2*t*(c2-c1)
        a_, b_, _ = MS.basis(tan, tuple(ref))
        if b_ @ up < 0: a_, b_ = -a_, -b_
        ring = []
        for k in range(sides):
            th = math.pi/2 + 2*math.pi*k/sides; c_, s_ = math.cos(th), math.sin(th)
            y = rh*fh*(1-abs(c_)**1.7) if s_ >= 0 else -0.34*rh*fh*(1-abs(c_)**2)
            ring.append(m.v(p + a_*rw*fw*c_ + b_*y))
        rings.append(ring)
    for i in range(nr-1):
        for k in range(sides):
            k2 = (k+1) % sides; m.f([rings[i][k], rings[i][k2], rings[i+1][k2], rings[i+1][k]], 'claw')
    c = m.v(c0)
    for k in range(sides): m.f([c, rings[0][(k+1) % sides], rings[0][k]], 'claw')
    c = m.v(c2)
    for k in range(sides): m.f([c, rings[-1][k], rings[-1][(k+1) % sides]], 'claw')
    return m


def band(P, q, a, w, R, fall=0.25, nd=None, soft=0.3):
    """Skin crease: 1 inside a thin slab through q (normal a, half-thickness ~w) within radius R of the axis,
    optionally only on the nd side of the digit."""
    r_ = P-q; t = r_ @ a
    g = np.exp(-(t/w)**2)
    rad = np.linalg.norm(r_-np.outer(t, a), axis=1)
    g = g*np.clip((R-rad)/fall+1, 0, 1)
    if nd is not None: g = g*np.clip(0.5+0.5*(r_ @ nd)/soft, 0, 1)
    return g


def evalcreases(P, cr):
    g = np.zeros(len(P))
    for q, a, w, R, nd, wt in cr: g = np.maximum(g, wt*band(P, q, a, w, R, nd=nd))
    return g


def decimate(verts, faces, target):
    """Quadric edge-collapse: keeps faces where the surface bends (fingers, creases), drops them on smooth stretches."""
    import pyfqmr
    if not target or len(faces) <= target: return verts, faces
    sp_ = pyfqmr.Simplify(); sp_.setMesh(np.asarray(verts, float), np.asarray(faces, np.int32))
    sp_.simplify_mesh(target_count=int(target), aggressiveness=5, preserve_border=True, verbose=False)
    v, f, _ = sp_.getMesh(); f = np.asarray(f, int)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return np.asarray(v, float), f


# ------------------------------------------------------------------------------------------------ hand
def pad_mesh(name, group, c, r, R, bend=0.0):
    """Palm pad. bend != 0 curls the two ends sideways, giving the bean shape in the drawings."""
    m = MS.ellipsoid(name, group, (0, 0, 0), r, 'pad', N=PAD_RES[0], lat=PAD_RES[1]); out = []
    for v in m.verts:
        v = np.array(v, float); v[1] += bend*r[1]*((v[0]/r[0])**2-0.35)
        out.append(np.array(c, float)+R @ v)
    m.verts = out
    return m


def make_hand(wr, el, s, hd, at, cl, curl=1.0, creases=True):
    """-> field, crease, claws, pads[(c, r, R, bend)], lo, hi

    Proportions are measured off the drawings, in wrist-widths (ww): palm ~1.4 ww long, fingers ~1.3 ww, visible claws
    ~0.5 ww (thumb claw ~0.8 ww), four fingers ~1.5 ww across, ~2.3 ww across with the thumb. Fingers are thick, touching,
    smoothly tapered (no knobby knuckles); two big round pads fill the heel of the palm.
    """
    H = hd*0.95; inn = -s
    d = nrm(V(inn*0.06, -1, -0.1))                 # down the hand
    n0 = nrm(V(inn*0.75, 0, 0.66))                 # palm faces the thigh and a little backwards
    w = nrm(np.cross(d, n0))
    if w[2] > 0: w = -w                            # +w = thumb side, pointing forward
    n = nrm(np.cross(w, d))
    if n @ n0 < 0: n = -n
    R = np.stack([d, w, n], 1)
    X = lambda u, v, h: wr + (u*d + v*w + h*n)*H
    fa = wr-el
    palm = [('cone', el+fa*0.8, wr, 1.47*at-0.1, 1.35*at-0.02, None),
            ('ell', X(0.7, 0.0, -0.02), (1.25*H, 1.3*H, 0.8*H), R, 0.8*H)]            # carpal block, as thick as the wrist
    claws = []; pts = [wr, el+fa*0.8]; cr = []
    Lp = 2.85
    #          v      len   u-fac spread curl+ radius
    fingers = [(1.14, 0.95, 0.96, 5, 0, 1.0), (0.38, 1.0, 1.0, 2, 2, 1.0), (-0.38, 0.96, 0.97, -2, 4, 0.97), (-1.14, 0.82, 0.88, -7, 7, 0.9)]
    for v, ln, uf, spd, ec, rs in fingers:
        palm.append(('cone', X(0.5, v*0.5, -0.05), X(Lp*uf, v, -0.06), 0.58*H, 0.47*H*rs, 0.5*H))         # metacarpal
    palm.append(('ell', X(1.55, 0.0, 0.2), (1.55*H, 1.5*H, 0.55*H), R, 0.4*H))                          # palm fill
    for v, ln, uf, spd, ec, rs in fingers:
        palm.append(('cone', X(0.3, v*0.35, -0.6), X(Lp*uf-0.1, v, -0.5), 0.16*H, 0.2*H, 0.18*H))        # extensor tendon
    # heel-of-palm pads: a soft bulge in the skin, the pad itself is its own mesh
    pads = [(X(1.2, 0.74, 0.85), (1.15*H, 0.74*H, 0.42*H), R, -0.42), (X(1.3, -0.74, 0.85), (1.2*H, 0.74*H, 0.42*H), R, 0.42)]
    for c, r, _, _ in pads: palm.append(('ell', c-n*0.1*H, tuple(x*0.88 for x in r), R, 0.2*H))
    groups = []
    rad = (0.45, 0.42, 0.38, 0.34)
    for v, ln, uf, spd, ec, rs in fingers:
        g = []; p = X(Lp*uf, v, -0.06); a_ = math.radians(spd); bd = math.cos(a_)*d + math.sin(a_)*w; ang = 0.0
        g.append(('sph', p-n*0.05*H, 0.48*rs*H, None))                                                  # knuckle (low, not knobby)
        cr.append((p+bd*0.12*H, bd, 0.07*H, 0.62*rs*H, -n, 0.6)); prev = bd
        for gi, (L, fx) in enumerate(zip((1.1, 0.85, 0.65), (10, 36, 36))):
            ang += math.radians(fx*curl+ec)
            dr = math.cos(ang)*bd + math.sin(ang)*n; pn = math.cos(ang)*n - math.sin(ang)*bd
            q = p + dr*L*ln*H
            g.append(('cone', p, q, rad[gi]*rs*H, rad[gi+1]*rs*H, 0.18*H))
            if gi < 2:
                g.append(('sph', q-pn*0.03*H, rad[gi+1]*1.05*rs*H, 0.15*H))
                R_ = rad[gi+1]*1.3*rs*H
                for o in (-0.11, 0.11): cr.append((q+dr*o*H, dr, 0.065*H, R_, -pn, 1.0))        # knuckle wrinkles on the back
                cr.append((q, dr, 0.065*H, R_, pn, 0.8))                                          # fold on the palm side
            p = q; pts.append(q)
        g.append(('sph', p+pn*0.08*H-dr*0.02*H, 0.37*rs*H, 0.1*H))                                     # fingertip bulb
        Lc = 1.55*cl*H*rs
        c0 = p-dr*0.3*H-pn*0.12*H
        claws.append((c0, c0+dr*0.66*Lc-pn*0.1*Lc, c0+dr*0.74*Lc+pn*0.66*Lc, 0.37*rs*H, 0.52*rs*H, w, -pn))
        pts.append(claws[-1][2]); groups.append((g, 0.35*H))
    # thumb: thick, set on the side of the wrist, with the biggest claw
    t0 = X(0.3, 0.7, 0.1); t1 = X(1.2, 1.75, 0.4)
    palm.append(('ell', X(0.95, 1.1, 0.2), (1.1*H, 0.85*H, 0.6*H), R, 0.35*H))                          # thenar mass
    g = [('cone', t0, t1, 0.74*H, 0.64*H, None)]
    td1 = nrm(0.8*d+0.15*w+0.55*n); t2 = t1+td1*1.15*H
    g += [('sph', t1, 0.66*H, 0.2*H), ('cone', t1, t2, 0.6*H, 0.5*H, 0.18*H)]
    td2 = nrm(0.7*d-0.25*w+0.65*n); t3 = t2+td2*0.8*H
    g += [('sph', t2, 0.52*H, 0.15*H), ('cone', t2, t3, 0.5*H, 0.42*H, 0.15*H), ('sph', t3, 0.44*H, 0.1*H)]
    pnt = nrm(n-(n @ td2)*td2); Lc = 1.8*cl*H
    c0 = t3-td2*0.32*H-pnt*0.12*H
    claws.append((c0, c0+td2*0.66*Lc-pnt*0.1*Lc, c0+td2*0.74*Lc+pnt*0.66*Lc, 0.44*H, 0.6*H, nrm(np.cross(td2, pnt)), -pnt))
    for q_, a_ in ((t1, td1), (t2, td2)): cr.append((q_, a_, 0.065*H, 0.75*H, None, 0.8))
    # wrist folds and the crease under the finger bases
    for u_ in (0.0, -0.24): cr.append((X(u_, 0, 0), d, 0.07*H, 1.7*at+0.3, n, 0.8))
    cr.append((X(Lp-0.5, 0, 0.3), d, 0.07*H, 1.75*H, n, 0.7))
    pts += [t1, t2, t3, claws[-1][2]]; groups.append((g, 0.45*H))

    crease = (lambda P: evalcreases(P, cr)) if creases else None

    def field(P):
        base = evalprims(P, palm); d_ = base
        for g, k in groups: d_ = np.minimum(d_, sm(base, evalprims(P, g), k))
        return d_+0.075*H*crease(P) if creases else d_
    pts = np.array(pts); m = 1.5*H
    return field, crease, claws, pads, pts.min(0)-m, pts.max(0)+m


# ------------------------------------------------------------------------------------------------ foot
def make_foot(an, kn, s, F, cl, lt, k3, toeout=8.0, creases=True):
    """-> field, crease, claws, lo, hi.  Built pointing straight ahead (-z), then turned out about the ankle."""
    ax = an[0]; inn = -s; sn = an-kn; lw = lt**0.5
    th = math.radians(toeout)*s; c_, s_ = math.cos(th), math.sin(th)
    piv = V(ax, 0, an[2])
    def out(p):                                   # straight-ahead frame -> world (turned out)
        q = np.array(p, float)-piv
        return piv+V(q[0]*c_-q[2]*s_, q[1], q[0]*s_+q[2]*c_)
    def inv(P):                                   # world -> straight-ahead frame
        q = P-piv
        return piv+np.stack([q[:, 0]*c_+q[:, 2]*s_, q[:, 1], -q[:, 0]*s_+q[:, 2]*c_], 1)
    stub = [('cone', kn+sn*0.86, an, 1.26*lt-0.45, 1.15*lt-0.02, None)]
    body = [('ell', an+V(0, 0.0, 0.05), (1.15*lt+0.09, 1.3, 1.15*lt+0.28), None, None)]                              # ankle
    for sd in (1, -1): body.append(('sph', an+V(sd*(1.15*lt-0.1), -0.45, 0.3), 0.5, 0.3))                      # ankle bones
    body.append(('cone', an+V(0, 1.8, 0.8), V(ax, 2.5, 2.5), 0.7, 0.85, 0.5))                           # Achilles tendon
    body.append(('ell', V(ax, 2.7, an[2]+0.1), (1.15, 1.7, 1.7), None, 0.6))                            # heel bone under the ankle
    body.append(('ell', V(ax, 1.62, 2.3), (1.3, 1.62, 1.55), None, 0.6))                                # heel
    body.append(('cone', an+V(0, -1.0, -0.5), V(ax, 1.85, -2.7*F), 1.5, 1.25, 0.7))                     # instep slope
    body.append(('ell', V(ax, 0.82, -2.3*F), (2.45*k3, 0.86, 3.4*F), None, 0.5))                         # sole
    DX = [x*k3 for x in (-2.25, -0.75, 0.75, 2.25)]; cr = []
    claws = []; pts = [kn+sn*0.86, V(ax, 0, 4.4), V(ax-3.6*k3, 0, 0), V(ax+3.6*k3, 0, 0)]
    KZS = [z*F for z in ((-4.45, -4.8, -4.8, -4.4) if s > 0 else (-4.4, -4.8, -4.8, -4.45))]
    for dx, KZ in zip(DX, KZS):
        body.append(('cone', V(ax+dx*0.35, 1.9, -1.2*F), V(ax+dx, 1.3, KZ), 1.0, 0.9, 0.5))             # metatarsal
    for dx, KZ in zip(DX, KZS):
        body.append(('cone', V(ax+dx*0.2, 3.3, -0.5), V(ax+dx, 2.2, KZ+0.25), 0.3, 0.27, 0.25))         # dorsal tendon
    groups = []
    for dx, KZ in zip(DX, KZS):
        K = V(ax+dx, 1.3, KZ)
        m = K+V(dx*0.14, -0.3, -1.6*F); tp = m+V(dx*0.08, -0.12, -1.05*F)
        up = V(0, 1, 0); a1 = nrm(m-K); a2 = nrm(tp-m)
        cr.append((K+a1*0.55, a1, 0.075, 1.25, up, 0.7))                                                  # web line at the toe base
        for o in (-0.13, 0.13): cr.append((m+a2*o, a2, 0.07, 1.15, up, 1.0))                              # toe-knuckle wrinkles
        g = [('sph', K+V(0, 0.22, 0), 0.98, None), ('cone', K, m, 0.9, 0.8, 0.16), ('sph', m+V(0, 0.12, 0), 0.85, 0.12),
             ('cone', m, tp, 0.8, 0.78, 0.14), ('sph', tp, 0.84, 0.1)]
        groups.append((g, 0.35))
        Lc = 2.6*cl*F**0.5
        claws.append((tp+V(0, 0.3, 0.0), tp+V(dx*0.03, 1.0, -0.3-0.5*Lc), tp+V(dx*0.06, -0.78, -0.6-0.8*Lc), 0.66, 0.84, V(1, 0, 0), V(0, 1, 0)))
    # inner toe, set back like a dewclaw
    hb = V(ax+inn*2.1*k3, 1.6, -1.6*F); hdr = nrm(V(inn*0.4, -0.1, -0.9)); hm = hb+hdr*1.7
    groups.append(([('cone', hb-hdr*1.2+V(-inn*0.5, 0, 0), hm, 0.95, 0.74, None), ('sph', hm, 0.78, 0.12)], 0.4))
    Lc = 1.9*cl
    claws.append((hm+V(0, 0.3, 0.1), hm+hdr*0.55*Lc+V(0, 0.55, 0), hm+hdr*Lc+V(0, -1.15, 0), 0.5, 0.62, nrm(np.cross(V(0, 1, 0), hdr)), V(0, 1, 0)))
    for c in claws: pts.append(out(c[2]))
    claws = [(out(c0), out(c1), out(c2), rw, rh, out(piv+ref)-piv, up_) for c0, c1, c2, rw, rh, ref, up_ in claws]
    # wrinkles across the front of the ankle and above the heel
    ia = nrm(V(0, -0.55, -0.83)); iu = nrm(V(0, 0.83, -0.55))
    for o in (0.9, 1.45, 2.0): cr.append((an+V(0, -0.6, -0.4)+ia*o, ia, 0.075, 1.75, iu, 0.75))
    for y_ in (0.5, 1.1): cr.append((an+V(0, y_, 1.2), V(0, 1, 0), 0.075, 1.35, V(0, 0, 1), 0.6))

    crease = (lambda P: evalcreases(inv(P), cr)) if creases else None

    def field(P):
        Q = inv(P)
        base = evalprims(Q, body); d_ = base
        for g, k in groups: d_ = np.minimum(d_, sm(base, evalprims(Q, g), k))
        if creases: d_ = d_+0.085*evalcreases(Q, cr)
        return np.maximum(d_, 0.03-P[:, 1])
    pts = np.array(pts); m = 1.7
    lo = pts.min(0)-m; hi = pts.max(0)+m; lo[1] = -0.35
    return field, crease, claws, lo, hi


# ------------------------------------------------------------------------------------------------ meshing
def taubin(Vv, F, iters=5, lam=0.5, mu=-0.53):
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


def sculpt(field, lo, hi, step, smooth=4, target=0, crease=None):
    """-> verts, faces, per-vertex ramp level 0..15 (float)"""
    xs = np.arange(lo[0], hi[0]+1e-6, step); ys = np.arange(lo[1], hi[1]+1e-6, step); zs = np.arange(lo[2], hi[2]+1e-6, step)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vol = field(P).reshape(X.shape)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(step, step, step), gradient_direction='ascent')
    verts = verts+np.array([xs[0], ys[0], zs[0]])
    verts = taubin(verts, faces, iters=smooth)
    verts, faces = decimate(verts, faces, target)
    vn = vnormals(verts, faces)
    if np.mean(field(verts+vn*0.12)) < np.mean(field(verts-vn*0.12)):
        faces = faces[:, ::-1].copy(); vn = -vn
    occ = np.zeros(len(verts))
    for d_, w_ in ((0.5, 1.0), (1.0, 0.6), (1.8, 0.4)):
        sd = field(verts+vn*d_)
        occ += w_*np.clip((d_-np.maximum(sd, 0))/d_, 0, 1)
    ao = np.clip(1-1.0*occ/2.0, 0, 1)
    shade_v = 0.44+0.44*ao+0.14*(0.5+0.5*vn[:, 1])+0.06*np.clip(vn @ nrm(V(-0.4, 0.5, -0.76)), 0, 1)
    vlv = (shade_v-0.58)/0.033
    if crease is not None: vlv = vlv-4.5*crease(verts)
    return verts, faces, np.clip(vlv, 0, 15)


def to_mesh(name, group, verts, faces, vlv):
    m = MS.Mesh(name, group); m.verts = [v.copy() for v in verts]
    m.faces = [([int(a), int(b), int(c)], 'skinramp') for a, b, c in faces]
    m.vlv = [float(x) for x in vlv]; m.flv = [int(round(x)) for x in vlv[faces].mean(1)]
    return m
