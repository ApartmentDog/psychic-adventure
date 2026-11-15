package com.enderling.client;

import com.enderling.mesh.MeshData;
import com.enderling.mesh.Meshes;
import com.enderling.mesh.Rig;
import com.mojang.blaze3d.vertex.PoseStack;
import net.minecraft.client.renderer.SubmitNodeCollector;
import net.minecraft.client.renderer.entity.EntityRenderer;
import net.minecraft.client.renderer.entity.EntityRendererProvider;
import net.minecraft.client.renderer.rendertype.RenderType;
import net.minecraft.client.renderer.rendertype.RenderTypes;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.resources.Identifier;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.LivingEntity;

/**
 * Draws the enderling meshes in place of the vanilla enderman model.
 * Generic over LivingEntity so it does not depend on where the EnderMan class lives in 26.x.
 */
public class EnderlingRenderer extends EntityRenderer<LivingEntity, EnderlingRenderState> {
	/** Beyond this many blocks the low-poly mesh is used. */
	private static final double FAR_DISTANCE = 20.0;

	private static final Identifier[] TEXTURES = new Identifier[Meshes.NAMES.length];

	static {
		for (int i = 0; i < TEXTURES.length; i++) {
			TEXTURES[i] = Identifier.fromNamespaceAndPath("enderling", "textures/entity/" + Meshes.NAMES[i] + ".png");
		}
	}

	public EnderlingRenderer(EntityRendererProvider.Context context) {
		super(context);
	}

	@Override
	public EnderlingRenderState createRenderState() {
		return new EnderlingRenderState();
	}

	@Override
	public void extractRenderState(LivingEntity entity, EnderlingRenderState state, float partialTick) {
		super.extractRenderState(entity, state, partialTick);
		state.bodyYaw = Mth.rotLerp(partialTick, entity.yBodyRotO, entity.yBodyRot);
		float head = Mth.rotLerp(partialTick, entity.yHeadRotO, entity.yHeadRot);
		state.headYaw = Mth.wrapDegrees(head - state.bodyYaw);
		state.headPitch = entity.getXRot(partialTick);
		state.walkPos = entity.walkAnimation.position(partialTick);
		state.walkSpeed = entity.walkAnimation.speed(partialTick);
		state.death = entity.deathTime > 0 ? Math.min(1f, (entity.deathTime + partialTick - 1f) / 20f) : 0f;
		// stable per-enderman pick: the same enderman keeps the same body every frame and across reloads
		state.variant = Math.floorMod(entity.getUUID().hashCode(), Meshes.NAMES.length);
	}

	@Override
	public void submit(EnderlingRenderState state, PoseStack poseStack, SubmitNodeCollector collector, CameraRenderState camera) {
		super.submit(state, poseStack, collector, camera); // name tag / leash

		double dx = state.x - camera.pos.x, dy = state.y - camera.pos.y, dz = state.z - camera.pos.z;
		boolean far = dx * dx + dy * dy + dz * dz > FAR_DISTANCE * FAR_DISTANCE;
		MeshData mesh = Meshes.get(state.variant, far);

		Rig.Params p = new Rig.Params();
		p.bodyYawDeg = state.bodyYaw;
		p.headYawDeg = state.headYaw;
		p.headPitchDeg = state.headPitch;
		p.walkPos = state.walkPos;
		p.walkSpeed = state.walkSpeed;
		p.age = state.ageInTicks;
		p.death = state.death;
		final float[][] mats = Rig.pose(mesh, p);

		RenderType type = RenderTypes.entityCutout(TEXTURES[state.variant]);
		final int light = state.lightCoords;
		collector.submitCustomGeometry(poseStack, type, (pose, consumer) -> MeshDrawer.draw(mesh, mats, pose, consumer, light));
	}
}
