// Updated Atlantean Engine Service

const { spawn } = require('child_process');

// Spawn Python script with unbuffered output
const process = spawn('python', ['atlantean_engine.py'], { stdio: 'pipe', env: { PYTHONUNBUFFERED: 1 }});

process.stdout.on('data', (data) => {
    console.log(`Received data: ${data.toString()}`);
});

process.stderr.on('data', (data) => {
    console.error(`Error: ${data.toString()}`);
});
