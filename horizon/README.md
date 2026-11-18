# Enderling for Horizon (Rust)

The Rust port of the Fabric mod, for [Horizon](https://playhorizon.dev), the Rust rewrite of Minecraft 26.3.
Horizon does not run Java mods, so this is a WebAssembly mod that registers its own renderer for
`minecraft:enderman` and draws the same four meshes with the same animation.

## Install

1. Download `enderling.hmod` from the `horizon-latest` release of this repo.
2. Put it in the `horizon-mods` folder of your Horizon instance (launcher: instance menu, "Open folder").
3. Add the line `mod.enderling:true` to `horizon-mods.txt`, beside `options.txt`.
4. Start the game. If it refuses the mod, the Mods screen says why in red.

## Differences from the Fabric version

- Horizon's quad call takes one UV rectangle per quad, not one UV per vertex, so each quad is textured
  with the closest rectangle to its four UVs. The shading ramp therefore steps per quad.
- The first two endermen drawn each frame use the detailed mesh, the rest the low-poly one
  (`NEAR_PER_FRAME` in `src/lib.rs`), because every quad is a call into the game.
- An enderman keeps its body for as long as the game runs; the choice comes from its network id, which
  changes between sessions.

## Build

`.github/workflows/horizon.yml` builds it with `cargo horizon-mod build --wasm` and publishes the release.
