package com.enderling.client;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;
import net.minecraft.world.entity.EntityTypes;

public class EnderlingClient implements ClientModInitializer {
	@Override
	public void onInitializeClient() {
		// Registering again for the vanilla enderman type replaces its renderer.
		EntityRendererRegistry.register(EntityTypes.ENDERMAN, EnderlingRenderer::new);
	}
}
