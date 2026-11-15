#!/usr/bin/env python3
"""bbmodel (free-format meshes + one cube) -> compact binary mesh + texture PNG for the Fabric mod.

Binary layout (little endian):
  char[4] "ENDM", int32 version(=1), float32 height_blocks
  int32 partCount
  per part:
    uint8 nameLen, utf8 name, int32 parentIndex (-1 = root),
    float32 pivot[3]            (blocks, rest pose, model space, +Y up, front = -Z)
    int32 vertexCount, int32 indexCount
    vertexCount * 8 float32     (x y z nx ny nz u v)   u,v normalised 0..1, v down
    indexCount  * int32         (QUADS: 4 indices each; triangles stored as degenerate quads a,b,c,c)
Units: blocks (1.0 = one block). The model is scaled so its total height == TARGET_HEIGHT blocks.
"""
import sys, json, struct, base64, io, os
import numpy as np
from PIL import Image

TARGET_HEIGHT = 2.9  # vanilla enderman height in blocks


def cube_faces(e):
    x0, y0, z0 = e['from']; x1, y1, z1 = e['to']
    C = lambda x, y, z: (x, y, z)
    # (TL, TR, BL, BR, normal) -- same corner convention as the preview renderer
    return {
        'north': (C(x1, y1, z0), C(x0, y1, z0), C(x1, y0, z0), C(x0, y0, z0), (0, 0, -1)),
        'south': (C(x0, y1, z1), C(x1, y1, z1), C(x0, y0, z1), C(x1, y0, z1), (0, 0, 1)),
        'east': (C(x1, y1, z1), C(x1, y1, z0), C(x1, y0, z1), C(x1, y0, z0), (1, 0, 0)),
        'west': (C(x0, y1, z0), C(x0, y1, z1), C(x0, y0, z0), C(x0, y0, z1), (-1, 0, 0)),
        'up': (C(x0, y1, z0), C(x1, y1, z0), C(x0, y1, z1), C(x1, y1, z1), (0, 1, 0)),
        'down': (C(x0, y0, z1), C(x1, y0, z1), C(x0, y0, z0), C(x1, y0, z0), (0, -1, 0)),
    }


def mesh_element(e, W, H):
    """-> (verts[N,8], idx list). Smooth vertex normals when smooth_shading is set."""
    vid = list(e['vertices'].keys())
    pos = {k: np.array(v, dtype=np.float64) for k, v in e['vertices'].items()}
    smooth = e.get('smooth_shading', False)
    # accumulate area-weighted normals per vertex id
    acc = {k: np.zeros(3) for k in vid}
    tris = []  # (face-normal, [(vid, uv)*3])
    for f in e['faces'].values():
        vs = f['vertices']
        if len(vs) < 3:
            continue
        for i in range(1, len(vs) - 1):
            tri = [vs[0], vs[i], vs[i + 1]]
            a, b, c = (pos[t] for t in tri)
            n = np.cross(b - a, c - a)
            for t in tri:
                acc[t] += n
            tris.append((n, [(t, tuple(f['uv'][t])) for t in tri]))
    out_v, seen = [], {}

    def emit(n, t, uv):
        ln = np.linalg.norm(n)
        fn = n / ln if ln > 1e-12 else np.array([0, 1.0, 0])
        if smooth:
            nn = acc[t]; l = np.linalg.norm(nn); nn = nn / l if l > 1e-12 else fn
            key = (t, uv)
        else:
            nn = fn; key = (t, uv, tuple(np.round(fn, 4)))
        if key not in seen:
            seen[key] = len(out_v)
            p = pos[t]
            out_v.append([p[0], p[1], p[2], nn[0], nn[1], nn[2], uv[0] / W, uv[1] / H])
        return seen[key]

    # pair triangles (p,q,r) + (p,r,d) sharing a reversed edge into quads (p,q,r,d)
    edge = {}
    for ti, (n, tri) in enumerate(tris):
        v3 = [t for t, _ in tri]
        for k in range(3):
            edge[(v3[k], v3[(k + 1) % 3])] = ti
    used = [False] * len(tris)
    out_i = []
    for ti, (n, tri) in enumerate(tris):
        if used[ti]:
            continue
        used[ti] = True
        v3 = [t for t, _ in tri]; uv3 = [u for _, u in tri]
        done = False
        for k in range(3):
            p_, q_, r_ = v3[k], v3[(k + 1) % 3], v3[(k + 2) % 3]
            up, uq, ur = uv3[k], uv3[(k + 1) % 3], uv3[(k + 2) % 3]
            nb = edge.get((p_, r_))
            if nb is None or used[nb] or nb == ti:
                continue
            nv = [t for t, _ in tris[nb][1]]; nuv = [u for _, u in tris[nb][1]]
            j = nv.index(p_)
            if nv[(j + 1) % 3] != r_:
                continue
            d_, ud = nv[(j + 2) % 3], nuv[(j + 2) % 3]
            used[nb] = True
            out_i += [emit(n, p_, up), emit(n, q_, uq), emit(n, r_, ur), emit(tris[nb][0], d_, ud)]
            done = True
            break
        if not done:
            i0, i1, i2 = (emit(n, t, u) for t, u in tri)
            out_i += [i0, i1, i2, i2]
    return out_v, out_i


def cube_element(e, W, H):
    out_v, out_i = [], []
    for name, (tl, tr, bl, br, nrm) in cube_faces(e).items():
        u0, v0, u1, v1 = e['faces'][name]['uv']
        base = len(out_v)
        for p, (u, v) in ((tl, (u0, v0)), (tr, (u1, v0)), (bl, (u0, v1)), (br, (u1, v1))):
            out_v.append([p[0], p[1], p[2], nrm[0], nrm[1], nrm[2], u / W, v / H])
        out_i += [base, base + 1, base + 3, base + 2]   # tl, tr, br, bl (perimeter order)
    return out_v, out_i


def convert(src, dst_bin, dst_png):
    m = json.load(open(src))
    W, H = m['resolution']['width'], m['resolution']['height']
    els = {e['uuid']: e for e in m['elements']}

    # texture
    tex = m['textures'][0]['source']
    Image.open(io.BytesIO(base64.b64decode(tex.split(',', 1)[1]))).convert('RGBA').save(dst_png)

    # parts (depth-first)
    parts = []

    def walk(group, parent):
        idx = len(parts)
        parts.append(dict(name=group['name'], parent=parent, pivot=group['origin'], elems=[]))
        for c in group['children']:
            if isinstance(c, str):
                parts[idx]['elems'].append(els[c])
            else:
                walk(c, idx)
    for g in m['outliner']:
        if isinstance(g, str):
            raise SystemExit('root-level element not in a group: ' + g)
        walk(g, -1)

    # bounds -> scale
    allv = []
    for p in parts:
        for e in p['elems']:
            if e['type'] == 'mesh':
                allv += list(e['vertices'].values())
            else:
                allv += [e['from'], e['to']]
    allv = np.array(allv)
    ymin, ymax = allv[:, 1].min(), allv[:, 1].max()
    scale = TARGET_HEIGHT / (ymax - ymin)
    # origin: x/z stay centred on the model origin, y so feet touch 0
    off = np.array([0.0, -ymin, 0.0])

    buf = io.BytesIO()
    buf.write(b'ENDM'); buf.write(struct.pack('<if', 1, TARGET_HEIGHT)); buf.write(struct.pack('<i', len(parts)))
    tv = ti = 0
    for p in parts:
        V, I = [], []
        for e in p['elems']:
            v, i = (mesh_element(e, W, H) if e['type'] == 'mesh' else cube_element(e, W, H))
            I += [x + len(V) for x in i]; V += v
        V = np.array(V, dtype=np.float32).reshape(-1, 8)
        if len(V):
            V[:, 0:3] = (V[:, 0:3] + off) * scale
        piv = (np.array(p['pivot']) + off) * scale
        nm = p['name'].encode()
        buf.write(struct.pack('<B', len(nm))); buf.write(nm)
        buf.write(struct.pack('<i', p['parent']))
        buf.write(struct.pack('<3f', *piv))
        buf.write(struct.pack('<ii', len(V), len(I)))
        buf.write(V.astype('<f4').tobytes())
        buf.write(np.array(I, dtype='<i4').tobytes())
        tv += len(V); ti += len(I)
    open(dst_bin, 'wb').write(buf.getvalue())
    return dict(parts=[(p['name'], p['parent']) for p in parts], verts=tv, quads=ti // 4, scale=scale, size=len(buf.getvalue()))


if __name__ == '__main__':
    srcdir, outdir = sys.argv[1], sys.argv[2]
    suffix = sys.argv[3] if len(sys.argv) > 3 else ''
    os.makedirs(outdir, exist_ok=True)
    for name in ('main', 'bruiser', 'lanky', 'ember'):
        r = convert(f'{srcdir}/enderling_{name}_sculpt.bbmodel', f'{outdir}/{name}{suffix}.bin', f'{outdir}/{name}{suffix}.png')
        print(name, 'parts', len(r['parts']), 'verts', r['verts'], 'quads', r['quads'], 'scale %.3f' % r['scale'], 'bytes', r['size'])
