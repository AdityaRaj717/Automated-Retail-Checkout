const express = require('express');
const http = require('http');
const { Server } = require("socket.io");
const cors = require('cors');

const app = express();
app.use(cors());

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*", // Allow React and Python to connect
    methods: ["GET", "POST"]
  }
});

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  // 1. Python -> React (Video & Data)
  socket.on('processed_frame', (data) => socket.broadcast.emit('live_feed', data));
  socket.on('cart_update', (cart) => socket.broadcast.emit('cart_sync', cart));

  // 2. React -> Python (Commands)
  socket.on('request_scan', () => io.emit('execute_scan'));

  // 3. Cart Actions (Add/Remove/Clear) — React -> Python
  socket.on('cart_action', (action) => {
    console.log("Action received:", action);
    io.emit('cart_action_trigger', action);
  });

  // 4. Active Learning — Cashier Confirmation Flow
  // Python -> React: prompt cashier to confirm uncertain item
  socket.on('cashier_prompt', (data) => {
    console.log("Cashier prompt:", data.suggested_name, `(${data.confidence}%)`);
    socket.broadcast.emit('cashier_prompt_ui', data);
  });

  // React -> Python: cashier's confirmation/correction response
  socket.on('cashier_response', (data) => {
    console.log("Cashier response:", data);
    io.emit('cashier_confirm', data);
  });

  socket.on('disconnect', () => console.log('Client disconnected'));
});

server.listen(3000, () => {
  console.log('🚀 ARC Bridge Server v2.0 running on port 3000');
});

