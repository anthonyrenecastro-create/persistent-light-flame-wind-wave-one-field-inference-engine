const express = require('express');
const path = require('path');
const { AtlanteanEngine } = require('../services/atlanteanEngine');

const app = express();
const port = 3000;

const engine = new AtlanteanEngine();
engine.start();

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'atlantean-ide')));

app.post('/infer', async (req, res) => {
    try {
        const { prompt } = req.body;
        const result = await engine.infer(prompt);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Atlantean Inference Engine web server running at http://localhost:${port}`);
});