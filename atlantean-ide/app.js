// PANEL DRAGGING

document.querySelectorAll(".panel-header").forEach(header => {
  let panel = header.parentElement

  header.onmousedown = (e) => {
    let offsetX = e.clientX - panel.offsetLeft
    let offsetY = e.clientY - panel.offsetTop

    function move(e){
      panel.style.position = "absolute"
      panel.style.left = (e.clientX - offsetX) + "px"
      panel.style.top = (e.clientY - offsetY) + "px"
    }

    document.addEventListener("mousemove", move)

    document.onmouseup = () => {
      document.removeEventListener("mousemove", move)
    }
  }
})


// WEBGL FIELD SIMULATION

const canvas = document.getElementById("glCanvas")
const gl = canvas.getContext("webgl")

canvas.width = 600
canvas.height = 400

const vertShaderSrc = `
attribute vec2 position;
varying vec2 vUv;
void main(){
  vUv = (position + 1.0) * 0.5;
  gl_Position = vec4(position,0,1);
}
`

const fragShaderSrc = `
precision mediump float;
uniform sampler2D fieldTex;
varying vec2 vUv;

void main(){
  vec4 field = texture2D(fieldTex, vUv);
  float v = field.r;
  vec3 color = mix(vec3(0.0,0.2,0.7), vec3(1.0,0.6,0.2), v);
  float glow = 0.2 + 0.8 * smoothstep(0.45, 0.55, v);
  gl_FragColor = vec4(color * glow, 1.0);
}
`

function compileShader(src,type){
  let s = gl.createShader(type)
  gl.shaderSource(s,src)
  gl.compileShader(s)
  return s
}

let vert = compileShader(vertShaderSrc,gl.VERTEX_SHADER)
let frag = compileShader(fragShaderSrc,gl.FRAGMENT_SHADER)

let program = gl.createProgram()

gl.attachShader(program,vert)
gl.attachShader(program,frag)
gl.linkProgram(program)

gl.useProgram(program)

let vertices = new Float32Array([
  -1,-1,
  1,-1,
  -1,1,
  1,1
])

let buffer = gl.createBuffer()

gl.bindBuffer(gl.ARRAY_BUFFER,buffer)
gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW)

let position = gl.getAttribLocation(program,"position")

gl.enableVertexAttribArray(position)
gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0)

let fieldTex = gl.createTexture()
gl.bindTexture(gl.TEXTURE_2D, fieldTex)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)

function updateFieldTexture(field){
  const H = field.length
  const W = field[0].length
  const data = new Uint8Array(W * H * 4)
  for(let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      const idx = (y * W + x) * 4
      const v = field[y][x]
      const r = Math.floor((v + 1) * 127.5) // assuming field is -1 to 1
      data[idx] = r
      data[idx+1] = r
      data[idx+2] = r
      data[idx+3] = 255
    }
  }
  gl.bindTexture(gl.TEXTURE_2D, fieldTex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, data)
}

function render(){
  gl.drawArrays(gl.TRIANGLE_STRIP,0,4)
  requestAnimationFrame(render)
}

render()


// BACKEND INTEGRATION

const consoleText = document.getElementById("consoleText")
const editorBox = document.getElementById("editorBox")

let useBackend = true  // Assume server provides backend

editorBox.addEventListener("keydown", async (e) => {
  if (e.key === "Enter" && e.ctrlKey) {
    const prompt = editorBox.value
    consoleText.textContent += "\n\n>> Executing prompt:\n" + prompt

    if (useBackend) {
      try {
        const response = await fetch('/infer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        })
        const res = await response.json()
        consoleText.textContent += "\n\n[Result] " + JSON.stringify(res, null, 2)
        if (res.fields && res.fields.length > 0) {
          updateFieldTexture(res.fields[0])
        }
      } catch (err) {
        consoleText.textContent += "\n\n[Error] " + err.message
        useBackend = false
      }
    } else {
      consoleText.textContent += "\n\n[Info] Backend not available; using demo visualization."
    }
  }
})
