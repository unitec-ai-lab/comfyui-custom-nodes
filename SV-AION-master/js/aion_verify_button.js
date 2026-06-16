import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "AION.VerifyKeyButton",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (!["AionThetaNode", "AionFluxPrompterNode", "AionFusionNode"].includes(nodeData.name)) {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            const node = this;
            let isVerifying = false;

            node.addWidget("button", "Verify API Key", null, function () {
                if (isVerifying) return;

                const apiKeyWidget = node.widgets.find(w => w.name === "gemini_api_key");
                const statusWidget = node.widgets.find(w => w.name === "api_key_status");

                if (!apiKeyWidget || !apiKeyWidget.value) {
                    if (statusWidget) {
                        statusWidget.value = "Not Verified - No key provided";
                    }
                    app.graph.setDirtyCanvas(true);
                    return;
                }

                isVerifying = true;
                if (statusWidget) {
                    statusWidget.value = "Verifying...";
                }
                app.graph.setDirtyCanvas(true);

                api.fetchApi("/aion/verify_key", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ api_key: apiKeyWidget.value })
                })
                .then(function (response) {
                    return response.json();
                })
                .then(function (data) {
                    if (statusWidget) {
                        statusWidget.value = data.status || "Not Verified";
                    }
                })
                .catch(function (error) {
                    if (statusWidget) {
                        statusWidget.value = "Not Verified - " + error.message;
                    }
                })
                .finally(function () {
                    isVerifying = false;
                    app.graph.setDirtyCanvas(true);
                });
            });

            node.setSize(node.computeSize());
        };
    }
});
