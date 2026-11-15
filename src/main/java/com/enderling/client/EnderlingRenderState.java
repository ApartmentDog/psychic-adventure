package com.enderling.client;

import net.minecraft.client.renderer.entity.state.EntityRenderState;

/** Everything the renderer needs, copied from the entity during extraction (render states must be self-contained). */
public class EnderlingRenderState extends EntityRenderState {
	public float bodyYaw;
	public float headYaw;
	public float headPitch;
	public float walkPos;
	public float walkSpeed;
	public float death;
	public int variant;
}
