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
                draw.quad(&sprite, corners, q.uv, tint);
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
