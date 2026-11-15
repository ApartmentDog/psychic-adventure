import math, random, json, uuid, base64, io, os, sys, string
import numpy as np
from PIL import Image, ImageDraw

def V(*a): return np.array(a, dtype=float)
def nrm(v):
    n = np.linalg.norm(v); return v/n if n > 1e-9 else v
def U(): return str(uuid.uuid4())
def hex2rgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
def shade(c, f): return tuple(max(0, min(255, int(v*f))) for v in c)
def mix(a, b, t): return tuple(int(a[i]*(1-t)+b[i]*t) for i in range(3))

class Mesh:
    def __init__(s, name, group):
        s.name, s.group, s.verts, s.faces, s.uuid = name, group, [], [], U()
    def v(s, p): s.verts.append(np.array(p, float)); return len(s.verts)-1
    def f(s, idx, mat): s.faces.append((list(idx), mat))

def basis(t, ref=(1, 0, 0)):
    t = nrm(t); r = np.array(ref, float)
    a = r - np.dot(r, t)*t
    if np.linalg.norm(a) < 1e-3:
        r = np.array([0, 0, 1.0]); a = r - np.dot(r, t)*t
    a = nrm(a); b = np.cross(t, a); return a, b, t

def crom(p0, p1, p2, p3, t):
    return 0.5*((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t*t + (-p0+3*p1-3*p2+p3)*t**3)

def resample(pts, radii, sub):
    P = [np.array(p, float) for p in pts]
    R = [np.array((r, r) if np.isscalar(r) else r, float) for r in radii]
    n = len(P)
    eP = [2*P[0]-P[1]] + P + [2*P[-1]-P[-2]]
    eR = [2*R[0]-R[1]] + R + [2*R[-1]-R[-2]]
    op, orr = [], []
    for i in range(n-1):
        for k in range(sub):
            t = k/sub
            op.append(crom(eP[i], eP[i+1], eP[i+2], eP[i+3], t))
            orr.append(np.maximum(crom(eR[i], eR[i+1], eR[i+2], eR[i+3], t), 0.02))
    op.append(P[-1]); orr.append(R[-1])
    return op, orr

def HI(N): return max(8, int(round(N*2.4/2))*2)

def loft(name, group, pts, radii, mat, N=8, caps=(True, True), ref=(1, 0, 0), sub=5):
    N = HI(N); m = Mesh(name, group)
    P, R = resample(pts, radii, sub); n = len(P); rings = []
    for i, p in enumerate(P):
        t = P[1]-P[0] if i == 0 else (P[-1]-P[-2] if i == n-1 else P[i+1]-P[i-1])
        a, b, t = basis(t, ref)
        r1, r2 = R[i]
        rings.append([m.v(p + a*r1*math.cos(2*math.pi*k/N) + b*r2*math.sin(2*math.pi*k/N)) for k in range(N)])
    for i in range(n-1):
        for k in range(N):
            k2 = (k+1) % N
            m.f([rings[i][k], rings[i][k2], rings[i+1][k2], rings[i+1][k]], mat)
    if caps[0]:
        c = m.v(P[0])
        for k in range(N): m.f([c, rings[0][(k+1) % N], rings[0][k]], mat)
    if caps[1]:
        c = m.v(P[-1])
        for k in range(N): m.f([c, rings[-1][k], rings[-1][(k+1) % N]], mat)
    return m

def rotmat(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def ellipsoid(name, group, c, rad, mat, tilt=(0, 0, 0), N=8, lat=4):
    m = Mesh(name, group); c = np.array(c, float); rx, ry, rz = rad; R = rotmat(*tilt)
    N = HI(N); lat = lat*2+2
    P = lambda l: c + R @ np.array(l, float)
    bot = m.v(P([0, -ry, 0])); rings = []
    for k in range(1, lat+1):
        phi = math.pi*k/(lat+1); y = -ry*math.cos(phi); s = math.sin(phi)
        rings.append([m.v(P([rx*s*math.cos(2*math.pi*j/N), y, -rz*s*math.sin(2*math.pi*j/N)])) for j in range(N)])
    top = m.v(P([0, ry, 0]))
    for j in range(N): m.f([bot, rings[0][(j+1) % N], rings[0][j]], mat)
    for i in range(lat-1):
        for j in range(N):
            j2 = (j+1) % N; m.f([rings[i][j], rings[i][j2], rings[i+1][j2], rings[i+1][j]], mat)
    for j in range(N): m.f([top, rings[-1][j], rings[-1][(j+1) % N]], mat)
    return m

def spike(name, group, base, tip, r, mat, curve=(0, 0, 0), N=5, sub=2):
    base, tip, curve = np.array(base, float), np.array(tip, float), np.array(curve, float)
    pts = [base+(tip-base)*s+curve*s*s for s in (0, 0.25, 0.5, 0.75, 1.0)]
    return loft(name, group, pts, [r, r*0.82, r*0.58, r*0.3, r*0.04], mat, N=N, sub=sub)

def fur_patch(out, rng, group, name, center, normal, radius, n, length):
    center = np.array(center, float); nn = nrm(np.array(normal, float)); a, b, _ = basis(nn)
    for i in range(int(n*1.6)):
        ang = rng.uniform(0, 2*math.pi); rr = radius*math.sqrt(rng.uniform(0.05, 1))
        base = center + a*rr*math.cos(ang) + b*rr*math.sin(ang)
        d = nrm(nn + (base-center)/max(radius, 1e-6)*0.55 + V(0, 0.25, 0))
        L = length*1.3*rng.uniform(0.7, 1.25)
        out.append(spike(f'{name}_{i}', group, base-nn*0.4, base+d*L, 0.5*rng.uniform(0.8, 1.15), 'fur',
                         curve=V(0, -0.25*L, 0), N=3, sub=1))

def build_model(sp, seed):
    rng = random.Random(seed); M = []
    h0, t, cw, ww, lt, at, al, fs, cl = (sp[k] for k in ('h0', 'torso', 'cw', 'ww', 'lt', 'at', 'al', 'fs', 'cl'))
    dl, pk, hs, nl = sp['dl'], sp['pk'], sp['hs'], sp['nl']; hd = sp['hand']
    Y = lambda o: h0+o*t
    rs = [(Y(2.0), ww*4.4, 3.4), (Y(6.0), ww*4.2, 3.3), (Y(10.0), cw*5.8, 3.8), (Y(13.5), cw*7.0, 4.1),
          (Y(16.2), cw*6.6, 3.5), (Y(17.4), cw*4.4, 2.9)]
    M.append(loft('torso', 'body', [V(0, y, 0.2) for y, _, _ in rs], [(rx, rz) for _, rx, rz in rs], 'skin', N=10))
    M.append(ellipsoid('pelvis', 'body', (0, h0+1.0, 0.3), (5.4*lt**0.4, 3.3, 3.9), 'skin', N=10))
    if sp['abs']:
        for k in range(3):
            for s in (1, -1):
                M.append(ellipsoid(f'abs{k}_{s}', 'body', (s*1.25*ww, Y(3.4+2.5*k), -3.0), (1.35*ww, 1.1*t, 0.9), 'skin', lat=3))
        for s in (1, -1):
            M.append(ellipsoid(f'oblique_{s}', 'body', (s*3.7*ww, Y(5.0), -1.2), (1.5, 2.8*t, 2.0), 'skin', tilt=(0, 0, s*10), lat=3))
    sx = 7.0*cw+1.2; shy = Y(15.0)
    for s in (1, -1):
        if pk >= 0.8: M.append(ellipsoid(f'pec_{s}', 'body', (s*3.4*cw, Y(13.0), -3.3), (3.7*cw*pk, 2.4*pk, 1.4), 'skin', tilt=(0, 0, -s*8), N=10))
        M.append(ellipsoid(f'lat_{s}', 'body', (s*4.2*cw, Y(11.5), 2.2), (2.6*cw, 3.6*t, 1.8), 'skin', lat=3))
        M.append(ellipsoid(f'trap_{s}', 'body', (s*3.4, Y(17.2), 0.5), (3.4, 1.5, 2.4), 'skin', lat=3))
    neck_top = Y(19.4)
    M.append(loft('neck', 'body', [V(0, Y(16.8), 0.2), V(0, neck_top, -0.5)], [(2.5, 2.4), (2.1, 2.1)], 'skin'))
    hb = neck_top-0.8; hz = -0.9
    head = dict(x0=-hs/2, x1=hs/2, y0=hb, y1=hb+hs, z0=hz-hs/2, z1=hz+hs/2)
    groups = {'body': ((0, h0, 0), None), 'head': ((0, neck_top, hz), 'body'),
              'tail': ((0, h0+1.2, 3.4), 'body')}
    tailtip = V(0, h0-0.2, 3.0+3.6*nl)
    M.append(loft('tail', 'tail', [V(0, h0+1.2, 3.0), V(0, h0+0.6, 3.0+2.0*nl), tailtip], [(1.9, 1.9), (1.4, 1.4), (0.7, 0.7)], 'skin'))
    furs = sp['fur']
    if 'back' in furs: fur_patch(M, rng, 'body', 'furback', V(0, Y(4.0), 3.2), V(0, 0.35, 1), 3.0*ww, 10, 3.6)
    if 'mane' in furs: fur_patch(M, rng, 'body', 'furmane', V(0, Y(17.3), 2.2), V(0, 0.5, 1), 3.8*cw, 12, 3.4)
    if 'chest' in furs: fur_patch(M, rng, 'body', 'furchest', V(0, Y(12.5), -4.3), V(0, 0.2, -1), 2.2*cw, 7, 1.7)
    if 'tailtip' in furs: fur_patch(M, rng, 'tail', 'furtail', tailtip, V(0, 0.1, 1), 0.9, 5, 2.6)
    if sp['hipspike']:
        for s in (1, -1):
            b = V(s*4.2, Y(3.0), 2.4)
            M.append(spike(f'hipspike_{s}', 'body', b, b+V(s*1.2, 6.5*t, 1.0), 1.15, 'claw', curve=V(s*0.3, 0.5, 2.6), N=6))
    for s, n in ((1, 'L'), (-1, 'R')):
        ga, gf, gh = f'arm_{n}', f'forearm_{n}', f'hand_{n}'
        sh = V(s*sx, shy, 0); el = sh+V(s*0.5, -10.5*al, 0.8); wr = el+V(s*0.3, -9.4*al, -1.4)
        groups[ga] = (tuple(sh), 'body'); groups[gf] = (tuple(el), ga); groups[gh] = (tuple(wr), gf)
        M.append(ellipsoid(f'delt_{n}', ga, sh, (3.0*dl, 3.3*dl, 3.0*dl), 'skin', N=10))
        M.append(loft(f'upperarm_{n}', ga, [sh+V(-s*0.6, 0.4, 0), sh+(el-sh)*0.3+V(0, 0, -0.1), sh+(el-sh)*0.65+V(0, 0, -0.3), el+V(0, 0.3, 0)],
                      [(2.6*at, 2.6*at), (3.0*at, 2.8*at), (2.5*at, 2.4*at), (1.9*at, 1.9*at)], 'skin', N=10, caps=(False, False)))
        M.append(ellipsoid(f'elbow_{n}', gf, el, (1.9*at, 1.9*at, 1.9*at), 'skin', lat=3))
        M.append(loft(f'forearm_{n}', gf, [el, el+(wr-el)*0.35+V(0, 0, -0.3), wr],
                      [(1.95*at, 1.95*at), (2.2*at, 2.05*at), (1.25*at, 1.2*at)], 'skin', N=10, caps=(False, True)))
        pc = wr+V(0, -1.9*hd, -0.2)
        M.append(ellipsoid(f'palm_{n}', gh, pc, (1.2*hd**0.6, 2.0*hd, 1.7*hd), 'skin'))
        M.append(ellipsoid(f'palmpad_{n}', gh, pc+V(-s*1.0*hd**0.6, -0.2, -0.1), (0.45, 1.2*hd, 0.9*hd), 'pad', lat=3))
        for i, (dz, ln) in enumerate(zip((1.1, 0.35, -0.4, -1.1), (0.85, 1.0, 0.95, 0.75))):
            b = pc+V(0, -1.7*hd, dz*hd)
            p1, p2 = b+V(0, -1.6*ln*hd, -0.35*hd), b+V(0, -3.0*ln*hd, -1.1*hd)
            M.append(loft(f'finger_{n}{i}', gh, [b, p1, p2], [0.62*hd**0.7, 0.5*hd**0.7, 0.42*hd**0.7], 'skin', N=6, caps=(False, True)))
            L = 2.6*cl
            M.append(spike(f'fclaw_{n}{i}', gh, p2, p2+nrm(V(0, -1, -0.35))*L, 0.46*hd**0.7, 'claw', curve=V(0, 0.15*L, -0.5*L), N=5))
        b = pc+V(-s*0.5, -1.0*hd, -1.55*hd); p2 = b+V(-s*0.2, -1.2*hd, -0.9*hd)
        M.append(loft(f'thumb_{n}', gh, [b, p2], [0.6*hd**0.7, 0.45*hd**0.7], 'skin', N=6, caps=(False, True)))
        M.append(spike(f'tclaw_{n}', gh, p2, p2+nrm(V(0, -1, -0.6))*2.0*cl, 0.42*hd**0.7, 'claw', curve=V(0, 0.1, -0.9), N=5))
        if 'forearm' in furs:
            fur_patch(M, rng, gf, f'furarm_{n}', el+(wr-el)*0.3+V(s*1.9, 0, 0.3), V(s, 0.1, 0.2), 1.3, 5, 2.0)
        if sp['elbowspike']:
            b = el+V(s*1.0, 0, 1.5)
            M.append(spike(f'elbowspike_{n}', gf, b, b+V(s*0.4, 1.2, 3.5), 0.9, 'claw', N=5))
        # legs
        gl, gs, gfoot = f'leg_{n}', f'shin_{n}', f'foot_{n}'
        k3 = lt**0.3
        hip = V(s*3.3*k3, h0, 0.3); kn = V(s*3.5*k3, 0.55*h0, -1.8); an = V(s*3.1*k3, 0.17*h0, 1.8)
        groups[gl] = (tuple(hip), None); groups[gs] = (tuple(kn), gl); groups[gfoot] = (tuple(an), gs)
        M.append(loft(f'thigh_{n}', gl, [V(s*3.0, h0+1.4, 0.3), hip, hip+(kn-hip)*0.45+V(s*0.1, 0, -0.3), kn+V(0, 0.8, 0)],
                      [(3.3*lt, 3.3*lt), (3.5*lt, 3.5*lt), (3.1*lt, 3.2*lt), (2.3*lt, 2.4*lt)], 'skin', N=10, caps=(False, False)))
        M.append(ellipsoid(f'knee_{n}', gs, kn, (2.2*lt, 2.2*lt, 2.2*lt), 'skin', lat=3))
        M.append(loft(f'shin_{n}', gs, [kn, kn+(an-kn)*0.4+V(s*0.1, 0, 0.6), an],
                      [(2.3*lt, 2.3*lt), (2.3*lt, 2.7*lt), (1.35*lt, 1.45*lt)], 'skin', N=10, caps=(False, False)))
        ax = an[0]
        M.append(loft(f'foot_{n}', gfoot, [an+V(0, 0.6, -0.2), V(ax, 2.4, 0.9), V(ax, 1.4, -3.0*fs), V(ax, 1.1, -5.6*fs)],
                      [(1.45, 1.45), (1.7, 1.4), (1.9, 1.05), (1.85, 0.95)], 'skin', N=8, caps=(False, True)))
        M.append(ellipsoid(f'footpad_{n}', gfoot, (ax, 0.55, -4.4*fs), (1.6, 0.45, 1.7*fs), 'pad', lat=3))
        for i, dx in enumerate((-1.35, 0, 1.35)):
            dx = dx*k3
            q0, q1, q2 = V(ax+dx*0.7, 1.15, -5.2*fs), V(ax+dx*1.0, 1.0, -7.4*fs), V(ax+dx*1.25, 1.2, -8.6*fs)
            M.append(loft(f'toe_{n}{i}', gfoot, [q0, q1, q2], [(0.85, 0.8), (0.8, 0.75), (0.7, 0.65)], 'skin', N=6, caps=(False, True)))
            L = 3.8*cl*fs
            M.append(spike(f'talon_{n}{i}', gfoot, q2, q2+nrm(V(dx*0.08, -0.08, -1))*L, 0.75, 'claw', curve=V(0, -0.18*L, 0), N=5))
        if 'heel' in furs:
            fur_patch(M, rng, gfoot, f'furheel_{n}', V(ax, 3.0, 2.4), V(0, 0.25, 1), 1.0, 4, 2.2)
    return M, head, groups

MATS = ['skin', 'claw', 'pad', 'fur']
LEV = [0.92, 0.96, 1.0, 1.04, 1.08]

def mat_colors(sp):
    sk = hex2rgb(sp['sk'])
    return {'skin': sk, 'skinramp': sk, 'claw': sp.get('clawcol', (18, 15, 24)), 'pad': shade(mix(sk, (60, 50, 110), 0.1), 0.82),
            'fur': mix(sk, (10, 8, 20), 0.5)}

def paint_head(sp, rng):
    """Face laid out from the drawing: two wide slits running from near the top down through the eyes, big half-lidded
    eyes low on the face (3/4 of the way down, nearly touching in the middle), brow wrinkles above them, and a tilted
    grin with a wedge of pointed teeth right under the eyes."""
    sk = hex2rgb(sp['sk']); dark = shade(sk, 0.32); slit = shade(sk, 0.64)
    def base(k):
        im = Image.new('RGB', (64, 64)); px = im.load()
        for y in range(64):
            for x in range(64): px[x, y] = shade(sk, k*(1+rng.uniform(-0.03, 0.03)))
        return im
    front = base(1.0); d = ImageDraw.Draw(front)
    style, brow = sp['eye'], sp['brow']
    SL = ((10, 21), (43, 54))
    for x0, x1 in SL: d.rectangle([x0, 10, x1, 56], fill=slit)
    #        outer-lid y, inner-lid y, lowest point y
    LID = {'sleepy': ((43.5, 46.5, 54.0),)*2, 'glare': ((42.0, 47.5, 54.0),)*2, 'wide': ((41.0, 42.0, 54.5),)*2,
           'smirk': ((44.5, 47.0, 53.5), (42.0, 44.5, 54.5))}.get(style, ((43.5, 46.5, 54.0),)*2)
    for k, (cx, col, (x0, x1)) in enumerate(((15.5, sp['eyeL'], SL[0]), (48.5, sp['eyeR'], SL[1]))):
        e = hex2rgb(col); inn = 1 if k == 0 else -1; yo, yi, yb = LID[k]
        xo, xi = cx-inn*14, cx+inn*14
        poly = [(xo, yo), (xi, yi), (cx+inn*11, yb-2.2), (cx+inn*5, yb), (cx-inn*3, yb-0.3), (cx-inn*10, yb-2.8), (cx-inn*13.5, yo+2.5)]
        mask = Image.new('L', (64, 64), 0); ImageDraw.Draw(mask).polygon(poly, fill=255)
        iris = Image.new('RGB', (64, 64), e); di = ImageDraw.Draw(iris)
        di.ellipse([cx-9, yo+2, cx+9, yb+3], fill=mix(e, (255, 255, 255), 0.22))            # lighter pool low in the eye
        di.rectangle([x0, 0, x1, 63], fill=shade(e, 0.66))                                   # the slit shows through, darker
        di.line([(cx-inn*7, yo+2.5), (cx-inn*2, yo+3.2)], fill=mix(e, (255, 255, 255), 0.9), width=1)   # highlight streak
        front.paste(iris, (0, 0), mask)
        d.line([(xo, yo), (cx-inn*13.5, yo+2.5), (cx-inn*10, yb-2.8), (cx-inn*3, yb-0.3), (cx+inn*5, yb), (cx+inn*11, yb-2.2), (xi, yi)], fill=shade(e, 0.4), width=1)  # lower rim
        d.line([(xo, yo), (xi, yi)], fill=dark, width=2)                                     # heavy upper lid
        if brow != 'none':                                                                   # brow wrinkles, pinched toward the centre
            bw = 2 if brow == 'heavy' else 1; bc = shade(sk, 0.7)
            d.line([(cx-inn*12, yo-7), (cx+inn*2, yo-6), (cx+inn*12, yi-4.5)], fill=bc, width=bw)
            d.line([(cx-inn*8, yo-10.5), (cx+inn*12, yi-8)], fill=bc, width=1)
    # grin: lip line tilted across the face, white wedge of teeth hanging from it
    y0, y1 = {'sleepy': (56.0, 60.0), 'smirk': (56.0, 60.0), 'glare': (57.5, 59.5), 'wide': (58.0, 59.0)}.get(style, (57, 60))
    mx0, mx1 = 8, 45; top = lambda x: y0+(y1-y0)*(x-mx0)/(mx1-mx0)
    tx0, tx1 = 11, 40; th = lambda x: 3.6-2.4*(x-tx0)/(tx1-tx0)                              # teeth get shorter along the grin
    d.polygon([(tx0, top(tx0)), (tx1, top(tx1)), (tx1, top(tx1)+th(tx1)), (tx0, top(tx0)+th(tx0))], fill=(236, 232, 222))
    x = tx0+5
    while x < tx1-1:                                                                         # notches between the points of the teeth
        b_ = top(x)+th(x)
        d.polygon([(x-1.2, b_+0.6), (x+1.2, b_+0.6), (x, b_-1.6)], fill=shade(sk, 0.95))
        x += 5 if x < 26 else 4
    d.line([(mx0, top(mx0)-0.6), (mx1, top(mx1)-0.6)], fill=dark, width=1)
    faces = {'north': front, 'up': base(1.15), 'south': base(0.8), 'east': base(0.9), 'west': base(0.9), 'down': base(0.65)}
    return faces

HEADPOS = {'north': (0, 8), 'south': (64, 8), 'east': (128, 8), 'west': (192, 8), 'up': (0, 72), 'down': (64, 72)}

def make_atlas(sp, rng):
    W = 256; img = Image.new('RGBA', (W, W), (0, 0, 0, 0)); colors = mat_colors(sp)
    cells = {}
    for mi, mname in enumerate(MATS):
        for li, f in enumerate(LEV):
            idx = mi*5+li; u0, v0 = (idx % 64)*4, (idx//64)*4; cells[(mname, li)] = (u0, v0)
            for y in range(4):
                for x in range(4): img.putpixel((u0+x, v0+y), shade(colors[mname], f)+(255,))
    for i in range(16):
        u0, v0 = i*4, 4; cells[('skinramp', i)] = (u0, v0)
        col = shade(colors['skin'], 0.58+i*0.033)
        for y in range(4):
            for x in range(4): img.putpixel((u0+x, v0+y), col+(255,))
    for f, im in paint_head(sp, rng).items(): img.paste(im, HEADPOS[f])
    return img, cells

def finalize(name, sp, M, head, groups, atlas, cells, rng):
    elems = []; cube_uuid = U()
    for idx, m in enumerate(M):
        idmap = {}
        used = set()
        def newid():
            while True:
                s = ''.join(rng.choice(string.ascii_lowercase+string.digits) for _ in range(4))
                if s not in used: used.add(s); return s
        verts = {}
        for i, p in enumerate(m.verts): idmap[i] = newid(); verts[idmap[i]] = [round(float(x), 3) for x in p]
        faces = {}
        lv = max(0, min(4, 2+rng.choice([-1, 0, 0, 1]))); m.level = lv
        for fi, (ids, mat) in enumerate(m.faces):
            u0, v0 = cells[(mat, m.flv[fi])] if hasattr(m, 'flv') else cells[(mat, lv)]
            corners = [(u0+1, v0+1), (u0+3, v0+1), (u0+3, v0+3), (u0+1, v0+3)]
            if mat == 'skinramp' and hasattr(m, 'vlv'):      # shade varies smoothly across the face: each corner looks up its own spot on the ramp strip
                uvs = {idmap[i]: [round(4*m.vlv[i]+2, 2), 6] for i in ids}
            else:
                uvs = {idmap[i]: list(corners[k]) for k, i in enumerate(ids)}
            faces[newid()] = {'uv': uvs, 'vertices': [idmap[i] for i in ids], 'texture': 0}
        m.json = {'name': m.name, 'type': 'mesh', 'origin': [0, 0, 0], 'rotation': [0, 0, 0], 'color': idx % 8, 'uuid': m.uuid,
                  'export': True, 'visibility': True, 'locked': False, 'render_order': 'default', 'allow_mirror_modeling': True,
                  'smooth_shading': True, 'vertices': verts, 'faces': faces}
        elems.append(m.json)
    # head cube with painted faces
    fu = {}
    for f, (x, y) in HEADPOS.items(): fu[f] = {'uv': [x, y, x+64, y+64], 'texture': 0}
    cx = (head['x0']+head['x1'])/2; cy = (head['y0']+head['y1'])/2; cz = (head['z0']+head['z1'])/2
    elems.append({'name': 'head', 'box_uv': False, 'rescale': False, 'locked': False, 'light_emission': 0, 'render_order': 'default',
                  'allow_mirror_modeling': True, 'from': [head['x0'], head['y0'], head['z0']], 'to': [head['x1'], head['y1'], head['z1']],
                  'autouv': 0, 'color': 3, 'inflate': 0, 'origin': [cx, cy, cz], 'rotation': [0, 0, 0], 'faces': fu, 'type': 'cube', 'uuid': cube_uuid})
    if sp['brow'] == 'heavy':
        bu = cells[('skin', 1)]; bf = {f: {'uv': [bu[0]+1, bu[1]+1, bu[0]+3, bu[1]+3], 'texture': 0} for f in ('north', 'south', 'east', 'west', 'up', 'down')}
        top = head['y1']
        bid = U()
        elems.append({'name': 'brow_ridge', 'box_uv': False, 'rescale': False, 'locked': False, 'light_emission': 0, 'render_order': 'default',
                      'allow_mirror_modeling': True, 'from': [head['x0']-0.2, top-0.6*(head['y1']-head['y0']), head['z0']-0.9], 'to': [head['x1']+0.2, top-0.49*(head['y1']-head['y0']), head['z0']],
                      'autouv': 0, 'color': 3, 'inflate': 0, 'origin': [cx, top-0.55*(head['y1']-head['y0']), head['z0']], 'rotation': [0, 0, 0], 'faces': bf, 'type': 'cube', 'uuid': bid})
        head['brow_uuid'] = bid
    # outliner
    nodes = {}
    for gname, (piv, parent) in groups.items():
        nodes[gname] = {'name': gname, 'origin': [round(float(x), 4) for x in piv], 'rotation': [0, 0, 0], 'color': 0, 'uuid': U(), 'export': True,
                        'isOpen': False, 'locked': False, 'visibility': True, 'autouv': 0, 'children': []}
    roots = []
    for gname, (piv, parent) in groups.items():
        (nodes[parent]['children'] if parent else roots).append(nodes[gname])
    for m in M: nodes[m.group]['children'].append(m.uuid)
    nodes['head']['children'] += [cube_uuid] + ([head['brow_uuid']] if 'brow_uuid' in head else [])
    buf = io.BytesIO(); atlas.save(buf, 'PNG')
    tex = {'path': '', 'name': name+'.png', 'folder': '', 'namespace': '', 'id': '0', 'particle': False, 'render_mode': 'normal',
           'render_sides': 'auto', 'frame_time': 1, 'frame_order_type': 'loop', 'frame_order': '', 'frame_interpolate': False,
           'visible': True, 'mode': 'bitmap', 'saved': False, 'uuid': U(), 'relative_path': '',
           'source': 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}
    return {'meta': {'format_version': '4.10', 'model_format': 'free', 'box_uv': False}, 'name': name, 'model_identifier': name,
            'visible_box': [1, 1, 0], 'variable_placeholders': '', 'timeline_setups': [], 'unhandled_root_fields': {},
            'resolution': {'width': 256, 'height': 256}, 'elements': elems, 'outliner': roots, 'textures': [tex]}

BASE = dict(h0=24, torso=1.0, cw=1.0, ww=1.0, lt=1.0, at=1.0, al=1.0, fs=1.25, cl=1.4, hand=1.35, dl=1.0, pk=1.0, hs=8, nl=1.0,
            abs=True, eye='sleepy', brow='normal', fur=[], hipspike=False, elbowspike=False)
VARIANTS = {
 'enderling_main_smooth': dict(BASE, sk='#2d2a36', eyeL='#9d4dff', eyeR='#9d4dff', lt=1.12, fur=['back', 'tailtip', 'heel'], hipspike=True),
 'enderling_bruiser_smooth': dict(BASE, sk='#6a615c', eyeL='#52f27a', eyeR='#52f27a', h0=21, torso=0.95, cw=1.25, ww=1.2, lt=1.35, at=1.4,
                                al=0.92, hs=8.5, dl=1.4, pk=1.35, fs=1.5, nl=1.2, hand=1.5, eye='glare', brow='heavy',
                                fur=['mane', 'back', 'forearm', 'heel', 'tailtip']),
 'enderling_lanky_smooth': dict(BASE, sk='#1c1d29', eyeL='#46e08a', eyeR='#ff4fa3', h0=29, torso=1.15, cw=0.78, ww=0.8, lt=0.7, at=0.65,
                              al=1.18, hs=7.5, dl=0.7, pk=0.6, fs=1.3, cl=1.9, nl=0.8, abs=False, eye='wide', brow='none',
                              fur=['back'], hipspike=True, elbowspike=True),
 'enderling_ember_smooth': dict(BASE, sk='#3a2f2c', eyeL='#ff8a1f', eyeR='#ff8a1f', h0=25, torso=1.05, cw=1.05, ww=0.95, dl=1.1, pk=1.1,
                              nl=1.5, lt=1.08, eye='smirk', brow='normal', fur=['chest', 'back', 'tailtip', 'heel', 'forearm']),
}

def build_all(outdir=None):
    res = {}
    for i, (name, sp) in enumerate(VARIANTS.items()):
        rng = random.Random(500+i)
        M, head, groups = build_model(sp, 500+i)
        atlas, cells = make_atlas(sp, rng)
        data = finalize(name, sp, M, head, groups, atlas, cells, rng)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, name+'.bbmodel'), 'w') as f: json.dump(data, f, separators=(',', ':'))
        res[name] = (sp, M, head, atlas, cells, data)
        nf = sum(len(m.faces) for m in M)
        print(name, 'meshes', len(M), 'faces', nf, 'height', round(head['y1'], 1))
    return res

if __name__ == '__main__':
    build_all(sys.argv[1])
