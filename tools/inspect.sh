#!/bin/bash
# Diagnostic: print the real 26.x class and method names the renderer depends on.
MC=""
for j in $(find ~/.gradle/caches/fabric-loom .gradle -name "*.jar" 2>/dev/null); do
  if unzip -l "$j" 2>/dev/null | grep -q "net/minecraft/client/renderer/entity/EntityRenderer.class"; then MC="$j"; break; fi
done
echo "MC=$MC"
[ -z "$MC" ] && { find ~/.gradle/caches/fabric-loom .gradle -name "*.jar" | head -40; exit 0; }
echo "=== class search ==="
unzip -l "$MC" | grep -o "net/minecraft[^ ]*\.class" | grep -E "CameraRenderState|/RenderTypes|/RenderType\.class|SubmitNodeCollector|EnderMan|Enderman|EntityRenderState\.class|LivingEntityRenderState\.class|OverlayTexture|EntityTypes?\.class|WalkAnimationState"
cls() { unzip -l "$MC" | grep -o "net/minecraft[^ ]*/$1\.class" | head -1 | sed 's/\.class$//; s#/#.#g'; }
for n in CameraRenderState RenderTypes SubmitNodeCollector EntityRenderer EntityRenderState EndermanRenderer EnderManRenderer OverlayTexture WalkAnimationState; do
  c=$(cls "$n"); [ -z "$c" ] && { echo "=== $n: not found ==="; continue; }
  echo "=== javap $c ==="; javap -cp "$MC" -protected "$c" 2>&1 | cut -c1-400
done
echo "=== SubmitNodeCollector nested ==="
for c in $(unzip -l "$MC" | grep -o "net/minecraft[^ ]*SubmitNodeCollector\$[A-Za-z]*\.class" | sed 's/\.class$//; s#/#.#g'); do javap -cp "$MC" -protected "$c" 2>&1 | cut -c1-400; done
echo "=== OrderedSubmitNodeCollector ==="
c=$(cls OrderedSubmitNodeCollector); [ -n "$c" ] && javap -cp "$MC" -protected "$c" 2>&1 | cut -c1-400
echo "=== VertexConsumer ==="
javap -cp "$MC" -protected com.mojang.blaze3d.vertex.VertexConsumer 2>&1 | cut -c1-300
echo "=== LivingEntity (rot/walk/death) ==="
javap -cp "$MC" -protected net.minecraft.world.entity.LivingEntity 2>&1 | grep -i -E "rot|walkAnim|deathTime" | cut -c1-200
echo "=== Entity (rot/uuid) ==="
javap -cp "$MC" -protected net.minecraft.world.entity.Entity 2>&1 | grep -i -E "getXRot|getYRot|getUUID|xRotO|yRotO" | cut -c1-200
echo "=== EntityTypes / EntityType ENDERMAN ==="
for n in EntityTypes EntityType; do c=$(cls "$n"); [ -n "$c" ] && javap -cp "$MC" -protected "$c" 2>&1 | grep -i -E "ENDERMAN|^public" | cut -c1-200; done
