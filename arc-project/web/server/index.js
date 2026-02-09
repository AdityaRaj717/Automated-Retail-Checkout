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

  // 1. Receive Frame from Python -> Send to React
  socket.on('processed_frame', (data) => {
    socket.broadcast.emit('live_feed', data);
  });

  // 2. Receive Cart Update from Python -> Send to React
  socket.on('cart_update', (cart) => {
    socket.broadcast.emit('cart_sync', cart);
  });

  // 3. NEW: Receive "Scan Command" from React -> Send to Python
  socket.on('request_scan', () => {
    console.log("Triggering Scan...");
    io.emit('execute_scan'); // Broadcast to Python
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

server.listen(3000, () => {
  console.log('🚀 ARC Bridge Server running on port 3000');
});
