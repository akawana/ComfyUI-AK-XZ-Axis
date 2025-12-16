import { app } from "../../../scripts/app.js";

const TAG = "[AKXZRangePrompts]";
const log = (...a) => console.log(TAG, ...a);
const warn = (...a) => console.warn(TAG, ...a);
const err = (...a) => console.error(TAG, ...a);

log("module loaded");

app.registerExtension({
  name: "AKXZRangePrompts.dynamicInputs",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    log("beforeRegisterNodeDef", nodeData?.name);

    if (nodeData?.name !== "AK XZ Batch Prompts") return;

    function findInputIndex(node, name) {
      if (!node || !Array.isArray(node.inputs)) return -1;
      return node.inputs.findIndex((i) => i && i.name === name);
    }

    function getInput(node, name) {
      const idx = findInputIndex(node, name);
      return idx >= 0 ? node.inputs[idx] : null;
    }

    function isConnected(inp) {
      return !!(inp && inp.link != null);
    }

    function pairConnected(node, i) {
      const a = getInput(node, `pos_${i}`);
      const b = getInput(node, `neg_${i}`);
      const ok = isConnected(a) && isConnected(b);
      log("pairConnected", { i, ok, modelLink: a?.link, conLink: b?.link });
      return ok;
    }

    function ensurePairExists(node, i) {
      const a = getInput(node, `pos_${i}`);
      const b = getInput(node, `neg_${i}`);
      if (a && b) return;

      // Determine types from existing inputs (fallback to strings)
      const modelType = getInput(node, "pos_0")?.type || "STRING";
      const conType = getInput(node, "neg_0")?.type || "STRING";

      if (!a) {
        node.addInput(`pos_${i}`, modelType);
        log("addInput", `pos_${i}`, modelType);
      }
      if (!b) {
        node.addInput(`neg_${i}`, conType);
        log("addInput", `neg_${i}`, conType);
      }
    }

    function removeTrailingUnconnectedPairs(node, minKeepPairs) {
      // Remove from the end only if both sockets are present and both are NOT connected.
      for (let i = 10; i >= minKeepPairs; i--) {
        const aIdx = findInputIndex(node, `pos_${i}`);
        const bIdx = findInputIndex(node, `neg_${i}`);
        if (aIdx < 0 || bIdx < 0) continue;

        const a = node.inputs[aIdx];
        const b = node.inputs[bIdx];

        if (isConnected(a) || isConnected(b)) continue;

        // Remove higher index first
        const idxs = [aIdx, bIdx].sort((x, y) => y - x);
        for (const idx of idxs) {
          try {
            node.removeInput(idx);
          } catch (e) {
            err("removeInput failed", { idx, name: node.inputs[idx]?.name }, e);
          }
        }
        log("removed trailing pair", i);
      }
    }

    function updateInputs(node, reason = "unknown") {
      try {
        log("updateInputs", { id: node?.id, reason });

        // Always ensure pair 0 exists (python required)
        ensurePairExists(node, 0);

        // Determine how many pairs should be visible/present
        let lastFull = -1;
        for (let i = 0; i <= 10; i++) {
          ensurePairExists(node, i); // needed for connection checks if user already had them
          const ok = pairConnected(node, i);
          if (ok) lastFull = i;
          else break;
        }

        // We keep pairs up to (lastFull + 1) so user can connect the next one.
        const keepUpTo = Math.min(10, Math.max(0, lastFull + 1));

        // Ensure pairs 0..keepUpTo exist
        for (let i = 0; i <= keepUpTo; i++) ensurePairExists(node, i);

        // Remove trailing pairs beyond keepUpTo if they are fully unconnected
        removeTrailingUnconnectedPairs(node, keepUpTo + 1);

        // Resize/redraw
        try {
          node.setSize(node.computeSize());
          app.graph.setDirtyCanvas(true, true);
        } catch (e) {
          err("resize failed", e);
        }

        // Debug dump
        const dump = (node.inputs || []).map((i) => ({
          name: i?.name,
          type: i?.type,
          link: i?.link,
        }));
        log("inputs after update", dump);
      } catch (e) {
        err("updateInputs failed", e);
      }
    }

    nodeType.prototype.onNodeCreated = function () {
      log("onNodeCreated", { id: this?.id, comfyClass: this?.comfyClass });
      this.serialize_widgets = true;

      // Start by removing all optional pairs so only pair 0 remains, unless already connected.
      // Remove from the end so indices stay valid.
      removeTrailingUnconnectedPairs(this, 1);

      updateInputs(this, "onNodeCreated");
      setTimeout(() => updateInputs(this, "onNodeCreated t=10ms"), 10);
      setTimeout(() => updateInputs(this, "onNodeCreated t=100ms"), 100);
    };

    const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (...args) {
      log("onConnectionsChange", { id: this?.id, args });
      const r = origOnConnectionsChange ? origOnConnectionsChange.apply(this, args) : undefined;
      setTimeout(() => updateInputs(this, "onConnectionsChange t=10ms"), 10);
      return r;
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      log("onConfigure", { id: this?.id, args });
      const r = origOnConfigure ? origOnConfigure.apply(this, args) : undefined;
      setTimeout(() => updateInputs(this, "onConfigure t=10ms"), 10);
      return r;
    };
  },
});
