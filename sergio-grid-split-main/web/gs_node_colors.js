import { app } from "../../scripts/app.js";

const GS_NODE_COLOR = "#1a5c5c";
const GS_NODE_BGCOLOR = "#0d3333";

app.registerExtension({
    name: "comfyui_grid_split.NodeColors",
    
    async nodeCreated(node) {
        if (node.comfyClass && node.comfyClass.startsWith("GS_")) {
            node.color = GS_NODE_COLOR;
            node.bgcolor = GS_NODE_BGCOLOR;
        }
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name && nodeData.name.startsWith("GS_")) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                this.color = GS_NODE_COLOR;
                this.bgcolor = GS_NODE_BGCOLOR;
                return result;
            };
        }
    }
});
