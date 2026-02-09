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

  // NEW: Pass cart actions (Add/Remove/Clear) to Python
  socket.on('cart_action', (action) => {
    console.log("Action received:", action);
    io.emit('cart_action_trigger', action);
  });

  socket.on('disconnect', () => console.log('Client disconnected'));
});

server.listen(3000, () => {
  console.log('🚀 ARC Bridge Server running on port 3000');
});
