//! The "ENDM" v1 binary mesh written by `tools/convert.py`. Units are blocks, +Y up, model front = -Z.

pub struct Quad {
    /// Vertex indices, rotated so they line up with `uv`'s rectangle corners.
    pub idx: [u32; 4],
    /// u0, v0, u1, v1 in sixteenths of the texture, as `BlockEntityDrawContext::quad` takes them.
    pub uv: [f32; 4],
}

pub struct Part {
    pub name: String,
    pub parent: i32,
    pub pivot: [f32; 3],
    /// x, y, z per vertex.
    pub pos: Vec<[f32; 3]>,
    pub quads: Vec<Quad>,
}

pub struct Mesh {
    pub parts: Vec<Part>,
    pub head: Option<usize>,
    pub tail: Option<usize>,
    pub arm: [Option<usize>; 2],
    pub fore: [Option<usize>; 2],
    pub leg: [Option<usize>; 2],
    pub shin: [Option<usize>; 2],
    pub foot: [Option<usize>; 2],
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("mesh file ends early")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Fits the rectangle `quad` can express to a quad's four corner UVs. The four corners of the
/// rectangle are (u0,v0) (u0,v1) (u1,v1) (u1,v0) in the order the corners are given, so the best
/// start corner of the four (a rotation keeps the winding) is the one whose least-squares
/// rectangle is closest.
fn fit(idx: [u32; 4], uv: &[[f32; 2]]) -> Quad {
    let mut best: Option<(f32, Quad)> = None;
    for s in 0..4 {
        let o = [idx[s], idx[(s + 1) % 4], idx[(s + 2) % 4], idx[(s + 3) % 4]];
        let c = o.map(|i| uv[i as usize]);
        let u0 = (c[0][0] + c[1][0]) / 2.0;
        let u1 = (c[2][0] + c[3][0]) / 2.0;
        let v0 = (c[0][1] + c[3][1]) / 2.0;
        let v1 = (c[1][1] + c[2][1]) / 2.0;
        let t = [[u0, v0], [u0, v1], [u1, v1], [u1, v0]];
        let err: f32 = (0..4)
            .map(|j| (c[j][0] - t[j][0]).powi(2) + (c[j][1] - t[j][1]).powi(2))
            .sum();
        if best.as_ref().is_none_or(|(e, _)| err < *e) {
            best = Some((err, Quad { idx: o, uv: [u0 * 16.0, v0 * 16.0, u1 * 16.0, v1 * 16.0] }));
        }
    }
    best.unwrap().1
}

impl Mesh {
    pub fn load(bytes: &[u8]) -> Result<Mesh, String> {
        let mut c = Cursor { b: bytes, at: 0 };
        if c.take(4)? != b"ENDM" {
            return Err("bad mesh magic".into());
        }
        let version = c.i32()?;
        if version != 1 {
            return Err(format!("unsupported mesh version {version}"));
        }
        let _height = c.f32()?;
        let n = c.i32()? as usize;
        let mut parts = Vec::with_capacity(n);
        for _ in 0..n {
            let len = c.take(1)?[0] as usize;
            let name = String::from_utf8_lossy(c.take(len)?).into_owned();
            let parent = c.i32()?;
            let pivot = [c.f32()?, c.f32()?, c.f32()?];
            let vc = c.i32()? as usize;
            let ni = c.i32()? as usize;
            let mut pos = Vec::with_capacity(vc);
            let mut uv = Vec::with_capacity(vc);
            for _ in 0..vc {
                let v: [f32; 8] = std::array::from_fn(|_| c.f32().unwrap_or(0.0));
                pos.push([v[0], v[1], v[2]]);
                uv.push([v[6], v[7]]);
            }
            let mut quads = Vec::with_capacity(ni / 4);
            for _ in 0..ni / 4 {
                let q = [c.i32()? as u32, c.i32()? as u32, c.i32()? as u32, c.i32()? as u32];
                if q.iter().any(|&i| i as usize >= vc) {
                    return Err("mesh index out of range".into());
                }
                quads.push(fit(q, &uv));
            }
            parts.push(Part { name, parent, pivot, pos, quads });
        }
        let find = |n: &str| parts.iter().position(|p| p.name == n);
        Ok(Mesh {
            head: find("head"),
            tail: find("tail"),
            arm: [find("arm_L"), find("arm_R")],
            fore: [find("forearm_L"), find("forearm_R")],
            leg: [find("leg_L"), find("leg_R")],
            shin: [find("shin_L"), find("shin_R")],
            foot: [find("foot_L"), find("foot_R")],
            parts,
        })
    }
}
