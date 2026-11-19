//! Enderling: replaces the enderman's model with one of four sculpted characters, picked per
//! enderman. A Rust port of the Fabric mod, for Horizon.
mod mesh;
mod rig;

use mesh::Mesh;
use mod_api::{
    BlockEntityDrawContext, EntityRenderState, EntityRenderer, Hooks, RegisterEntityRenderersEvent,
    RegistrationContext,
};
use std::sync::{Arc, Mutex, OnceLock};

const NAMES: [&str; 4] = ["main", "bruiser", "lanky", "ember"];

macro_rules! mesh_bytes {
    ($($n:literal),*) => { [$( (include_bytes!(concat!("../meshes/", $n, ".bin")) as &[u8], include_bytes!(concat!("../meshes/", $n, "_lod.bin")) as &[u8]) ),*] };
}
/// (near, far) per variant, as shipped.
static BYTES: [(&[u8], &[u8]); 4] = mesh_bytes!("main", "bruiser", "lanky", "ember");
static LOADED: [[OnceLock<Result<Mesh, String>>; 2]; 4] = [const { [OnceLock::new(), OnceLock::new()] }; 4];

fn mesh(variant: usize, far: bool) -> Result<&'static Mesh, String> {
    let slot = &LOADED[variant][far as usize];
    let bytes = if far { BYTES[variant].1 } else { BYTES[variant].0 };
    slot.get_or_init(|| Mesh::load(bytes)).as_ref().map_err(|e| e.clone())
}

/// How many endermen per frame get the detailed mesh; the rest draw the low-poly one. Each quad
/// is one call into the game, so a crowd (the End) would cost a frame otherwise.
const NEAR_PER_FRAME: u32 = 2;

#[derive(Default)]
struct Frame {
    stamp: u64,
    first: u32,
    count: u32,
}

#[derive(Default)]
struct Enderling {
    frame: Mutex<Frame>,
}

impl Enderling {
    /// Whether this enderman is one of the first few drawn in the current frame.
    fn near(&self, id: u32, stamp: u64) -> bool {
        let mut f = self.frame.lock().unwrap();
        if f.stamp != stamp || f.first == id {
            *f = Frame { stamp, first: id, count: 0 };
        }
        f.count += 1;
        f.count <= NEAR_PER_FRAME
    }
}

/// A stable pick per enderman for as long as the game runs.
fn variant_of(id: u32) -> usize {
    let mut h = id.wrapping_mul(0x9E37_79B1);
    h ^= h >> 15;
    (h as usize) % NAMES.len()
}

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: V3) -> Option<V3> {
    let l = dot(a, a).sqrt();
    (l > 1e-9).then(|| [a[0] / l, a[1] / l, a[2] / l])
}
/// Rotates `v` by the 3x3 part of a pose matrix.
fn turn(m: &rig::M, v: V3) -> V3 {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
        m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
    ]
}

/// The brightness the game gives a face by the way it points (its `diffuse`, the overworld's two
/// lights with 0.4 ambient). It lights every quad by the quad's own flat normal.
fn diffuse(n: V3) -> f32 {
    const L0: V3 = [0.161_7, 0.808_4, -0.565_9];
    const L1: V3 = [-0.161_7, 0.808_4, 0.565_9];
    ((dot(L0, n).max(0.0) + dot(L1, n).max(0.0)) * 0.6 + 0.4).min(1.0)
}

/// What to multiply a quad's tint by so the game's flat lighting of it comes out as the lighting
/// of its smooth normal `smooth` (the Fabric version lights by vertex normals, which blend across
/// the sculpt; flat quads would show every facet). A tint can only darken, so a quad the flat
/// normal lights darker than the smooth one stays as it is.
fn relight(corners: &[V3; 4], smooth: V3) -> f32 {
    // As the game takes it: the first triangle's normal, else the second's.
    let Some(flat) = unit(cross(sub(corners[1], corners[0]), sub(corners[2], corners[0])))
        .or_else(|| unit(cross(sub(corners[2], corners[1]), sub(corners[3], corners[1]))))
    else {
        return 1.0;
    };
    let Some(smooth) = unit(smooth) else { return 1.0 };
    (diffuse(smooth) / diffuse(flat)).min(1.0)
}

fn scaled(rgb: u32, k: f32) -> u32 {
    let c = |shift: u32| (((rgb >> shift) & 255) as f32 * k).round() as u32;
    c(16) << 16 | c(8) << 8 | c(0)
}

impl EntityRenderer for Enderling {
    fn submit(
        &self,
        entity: &EntityRenderState,
        draw: &mut dyn BlockEntityDrawContext,
    ) -> Result<(), String> {
        let living = entity.living.ok_or("not a living entity")?;
        let variant = variant_of(entity.id);
        let near = self.near(entity.id, draw.animation_time().to_bits());
        let mesh = mesh(variant, !near)?;

        let death = if living.death_time > 0.0 { ((living.death_time - 1.0) / 20.0).min(1.0) } else { 0.0 };
        let mats = rig::pose(
            mesh,
            &rig::Params {
                body_yaw_deg: living.body_rot,
                head_yaw_deg: living.y_rot,
                head_pitch_deg: living.x_rot,
                walk_pos: living.walk_animation_pos,
                walk_speed: living.walk_animation_speed,
                age: entity.age,
                death,
            },
        );

        let sprite = format!("enderling:entity/{}", NAMES[variant]);
        let tint = if living.has_red_overlay { 0xFF9A9A } else { 0xFFFFFF };
        let scale = living.scale;
        let mut world: Vec<[f32; 3]> = Vec::new();
        for (part, m) in mesh.parts.iter().zip(&mats) {
            world.clear();
            world.extend(part.pos.iter().map(|p| {
                [
                    (m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3]) * scale,
                    (m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7]) * scale,
                    (m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]) * scale,
                ]
            }));
            for q in &part.quads {
                let corners = q.idx.map(|i| world[i as usize]);
                let k = relight(&corners, turn(m, q.n));
                draw.quad(&sprite, corners, q.uv, scaled(tint, k));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct Mod;

#[mod_api::horizon_mod]
impl Hooks for Mod {
    fn register_entity_renderers(
        &self,
        _: &RegistrationContext,
        event: &mut RegisterEntityRenderersEvent<'_>,
    ) {
        event.renderers.register("minecraft:enderman".into(), Arc::new(Enderling::default()));
    }
}
