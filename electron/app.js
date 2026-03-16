const runBtn = document.getElementById("runBtn")
const outputBox = document.getElementById("outputBox")

const canvas = document.getElementById("fieldCanvas")
const ctx = canvas.getContext("2d")

canvas.width = 500
canvas.height = 300

const { AtlanteanEngine } = require('../services/atlanteanEngine.js');
const engine = new AtlanteanEngine();

engine.start();

function drawField(fields) {
    if (!fields || !fields[0]) return;

    const field = fields[0]; // visualize field 0
    const H = field.length;
    const W = field[0].length;

    const scaleX = canvas.width / W;
    const scaleY = canvas.height / H;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for(let i = 0; i < H; i++){
        for(let j = 0; j < W; j++){
            let value = field[i][j];
            // normalize assuming values around -1 to 1
            value = (value + 1) / 2;
            value = Math.max(0, Math.min(1, value));
            // color: blue to red gradient
            const r = Math.floor(value * 255);
            const g = Math.floor(value * 100);
            const b = Math.floor(255 - value * 255);
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(j * scaleX, i * scaleY, scaleX, scaleY);
        }
    }
}

runBtn.onclick = async () => {

const prompt = document.getElementById("promptBox").value

if (!prompt) {
    outputBox.textContent = "Error: Empty prompt"
    return
}

runBtn.disabled = true

outputBox.textContent = `Processing prompt: "${prompt}"\n\nRunning inference...`

try {
    const result = await engine.infer(prompt)
    outputBox.textContent = `Prompt: "${prompt}"\n\nResult:\n${JSON.stringify({value_field3_mean: result.value_field3_mean, dodec_frac: result.dodec_frac, result: result.result, time_s: result.time_s}, null, 2)}`
    drawField(result.fields)  // Visualize actual field data
} catch (e) {
    outputBox.textContent = `Error: ${e.message}`
} finally {
    runBtn.disabled = false
}

}