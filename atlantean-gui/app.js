const runBtn = document.getElementById("runBtn")
const outputBox = document.getElementById("outputBox")

const canvas = document.getElementById("fieldCanvas")
const ctx = canvas.getContext("2d")

canvas.width = 500
canvas.height = 300

function drawField() {

const w = canvas.width
const h = canvas.height

for(let x=0;x<w;x+=4){
for(let y=0;y<h;y+=4){

let value = Math.random()

ctx.fillStyle = `rgb(${value*255},${value*150},255)`
ctx.fillRect(x,y,4,4)

}
}

}

runBtn.onclick = () => {

const prompt = document.getElementById("promptBox").value

outputBox.textContent =
`Prompt received:

"${prompt}"

(No backend connected)

Visualization running...`

drawField()

}