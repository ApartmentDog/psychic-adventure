//! Procedural skeleton animation: one 3x4 row-major matrix per part (a port of `Rig.java`).
use crate::mesh::Mesh;

pub struct Params {
    pub body_yaw_deg: f32,
    pub head_yaw_deg: f32,
    /// Positive looks down.
    pub head_pitch_deg: f32,
    pub walk_pos: f32,
    pub walk_speed: f32,
    pub age: f32,
    /// 0..1, falling over sideways.
    pub death: f32,
}

pub type M = [f32; 12];

fn rot_x(a: f32) -> M {
    let (s, c) = a.sin_cos();
    [1., 0., 0., 0., 0., c, -s, 0., 0., s, c, 0.]
}
fn rot_y(a: f32) -> M {
    let (s, c) = a.sin_cos();
    [c, 0., s, 0., 0., 1., 0., 0., -s, 0., c, 0.]
}
fn rot_z(a: f32) -> M {
    let (s, c) = a.sin_cos();
    [c, -s, 0., 0., s, c, 0., 0., 0., 0., 1., 0.]
}

/// a * b for 3x4 affine matrices.
fn mul(a: &M, b: &M) -> M {
    let mut o = [0.0; 12];
    for r in 0..3 {
        for c in 0..3 {
            o[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c];
        }
        o[r * 4 + 3] = a[r * 4] * b[3] + a[r * 4 + 1] * b[7] + a[r * 4 + 2] * b[11] + a[r * 4 + 3];
    }
    o
}

fn set(a: &mut [f32], i: Option<usize>, v: f32) {
    if let Some(i) = i {
        a[i] = v;
    }
}

pub fn pose(m: &Mesh, p: &Params) -> Vec<M> {
    let n = m.parts.len();
    let mut ang_x = vec![0.0f32; n];
    let mut ang_y = vec![0.0f32; n];

    let walk = p.walk_speed.clamp(0.0, 1.0);
    let swing = (p.walk_pos * 0.6662).cos() * walk;
    let sway = (p.age * 0.067).sin() * 0.04;

    let leg_l = 0.75 * swing;
    let leg_r = -leg_l;
    let arm_l = -0.6 * swing + sway;
    let arm_r = 0.6 * swing - sway;

    set(&mut ang_x, m.leg[0], leg_l);
    set(&mut ang_x, m.leg[1], leg_r);
    set(&mut ang_x, m.shin[0], -(0.06 + 1.1 * (-leg_l).max(0.0)));
    set(&mut ang_x, m.shin[1], -(0.06 + 1.1 * (-leg_r).max(0.0)));
    for k in 0..2 {
        if let (Some(f), Some(s)) = (m.foot[k], m.shin[k]) {
            ang_x[f] = -ang_x[s] * 0.6;
        }
    }
    set(&mut ang_x, m.arm[0], arm_l);
    set(&mut ang_x, m.arm[1], arm_r);
    set(&mut ang_x, m.fore[0], 0.12 + 0.6 * arm_l.max(0.0));
    set(&mut ang_x, m.fore[1], 0.12 + 0.6 * arm_r.max(0.0));

    let hy = p.head_yaw_deg.clamp(-75.0, 75.0);
    let hp = p.head_pitch_deg.clamp(-60.0, 60.0);
    if let Some(h) = m.head {
        ang_y[h] = -hy.to_radians();
        ang_x[h] = -hp.to_radians();
    }
    if let Some(t) = m.tail {
        ang_y[t] = (p.walk_pos * 0.3331).sin() * 0.3 * walk + (p.age * 0.05).sin() * 0.08;
    }

    // root: face the right way, optionally fall over (roll about the model's Z axis)
    let mut root = mul(&rot_y((180.0 - p.body_yaw_deg).to_radians()), &rot_z((90.0 * p.death).to_radians()));
    root[7] += 0.6 * p.death;

    let mut out: Vec<M> = Vec::with_capacity(n);
    for i in 0..n {
        let part = &m.parts[i];
        let base = if part.parent < 0 { root } else { out[part.parent as usize] };
        if ang_x[i] == 0.0 && ang_y[i] == 0.0 {
            out.push(base);
            continue;
        }
        let mut r = mul(&rot_y(ang_y[i]), &rot_x(ang_x[i]));
        let pv = part.pivot;
        r[3] = pv[0] - (r[0] * pv[0] + r[1] * pv[1] + r[2] * pv[2]);
        r[7] = pv[1] - (r[4] * pv[0] + r[5] * pv[1] + r[6] * pv[2]);
        r[11] = pv[2] - (r[8] * pv[0] + r[9] * pv[1] + r[10] * pv[2]);
        out.push(mul(&base, &r));
    }
    out
}
