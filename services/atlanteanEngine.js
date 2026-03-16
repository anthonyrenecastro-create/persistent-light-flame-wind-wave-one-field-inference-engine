const { spawn } = require("child_process")
const path = require("path")

class AtlanteanEngine {

    start(){
        // Use -u to force unbuffered stdout/stderr from Python so Electron IPC works reliably
        const py = process.platform === "win32" ? "python" : "python3"
        this.process = spawn(py, ["-u", path.join(__dirname, "..", "atlantean", "atlantean_engine.py")], {
            stdio: ["pipe", "pipe", "pipe"]
        })

        this.process.stderr.on("data", (data) => {
            console.error("[atlantean-engine stderr]", data.toString())
        })

        this.process.on("exit", (code, sig) => {
            console.warn(`[atlantean-engine] exited code=${code} sig=${sig}`)
        })
    }

    infer(prompt){
        return new Promise((resolve, reject)=>{

            if (!this.process || this.process.killed) {
                return reject(new Error("Engine process not running"))
            }

            // send prompt
            this.process.stdin.write(prompt + "\n")

            // read a single JSON line response
            const onData = (data) => {
                try {
                    const s = data.toString()
                    // try to parse JSON; if parse fails, return raw string
                    try {
                        const j = JSON.parse(s)
                        resolve(j)
                    } catch (e) {
                        resolve(s)
                    }
                } finally {
                    // remove listener (we used once)
                    this.process.stdout.removeListener("data", onData)
                }
            }

            this.process.stdout.once("data", onData)

            // also guard for errors
            this.process.once("error", (err) => {
                reject(err)
            })

        })

    }

}

module.exports = { AtlanteanEngine }