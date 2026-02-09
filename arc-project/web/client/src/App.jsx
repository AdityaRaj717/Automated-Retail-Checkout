import React, { useEffect, useState, useRef } from 'react';
import io from 'socket.io-client';
import { ShoppingBag, X, CreditCard, Scan, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const socket = io('http://localhost:3000');

function App() {
  const [imageSrc, setImageSrc] = useState(null);
  const [cart, setCart] = useState([]);
  const [total, setTotal] = useState(0);
  const [isScanning, setIsScanning] = useState(false);
  const lastItemRef = useRef(null);

  // Trigger scan on Spacebar
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault(); // Prevent scrolling
        console.log("Space pressed - Requesting scan");
        socket.emit('request_scan');
        setIsScanning(true);
        setTimeout(() => setIsScanning(false), 2000); // Visual feedback
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    socket.on('live_feed', (data) => {
      setImageSrc(`data:image/jpeg;base64,${data.image}`);
      if (data.image) setIsScanning(true);
    });

    socket.on('cart_sync', (newCart) => {
      setCart(newCart);
      const newTotal = newCart.reduce((sum, item) => sum + item.price, 0);
      setTotal(newTotal);
    });

    return () => socket.off();
  }, []);

  // Auto-scroll to bottom of cart
  useEffect(() => {
    lastItemRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [cart]);

  return (
    <div className="min-h-screen bg-black text-white font-sans overflow-hidden flex">

      {/* LEFT: Immersive Camera Feed */}
      <div className="relative flex-1 bg-gray-900 overflow-hidden">
        {imageSrc ? (
          <img
            src={imageSrc}
            className="w-full h-full object-cover opacity-90"
            alt="Live Stream"
          />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-4">
            <div className="relative w-16 h-16">
              <span className="absolute inset-0 border-t-2 border-l-2 border-white rounded-tl-lg"></span>
              <span className="absolute inset-0 border-b-2 border-r-2 border-white rounded-br-lg animate-pulse"></span>
            </div>
            <p className="tracking-widest text-xs uppercase">System Offline</p>
          </div>
        )}

        {/* HUD Overlay */}
        <div className="absolute inset-0 pointer-events-none p-8 flex flex-col justify-between">
          <div className="flex items-center gap-4">
            <div className="h-2 w-2 bg-red-500 rounded-full animate-ping" />
            <span className="text-xs font-mono text-red-400 tracking-widest">LIVE FEED • REC</span>
          </div>

          {/* Scanning Box Graphic */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-[400px] h-[400px] border border-white/20 rounded-3xl relative">
              <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-blue-500 -mt-1 -ml-1" />
              <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-blue-500 -mt-1 -mr-1" />
              <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-blue-500 -mb-1 -ml-1" />
              <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-blue-500 -mb-1 -mr-1" />

              {isScanning && (
                <motion.div
                  initial={{ top: 0 }}
                  animate={{ top: "100%" }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="absolute left-0 right-0 h-0.5 bg-blue-500/50 shadow-[0_0_15px_rgba(59,130,246,0.6)]"
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT: Glassmorphism Cart */}
      <div className="w-[450px] bg-gray-900/90 backdrop-blur-2xl border-l border-white/10 flex flex-col relative z-10 shadow-2xl">

        {/* Header */}
        <div className="p-8 pb-4">
          <div className="flex items-center justify-between mb-2">
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Zap className="text-yellow-400 fill-yellow-400" size={24} />
              FastCheckout
            </h1>
            <span className="text-xs font-mono text-gray-500">{new Date().toLocaleTimeString()}</span>
          </div>
          <div className="h-1 w-full bg-gray-800 rounded-full mt-4 overflow-hidden">
            <div className="h-full bg-blue-500 w-2/3 animate-pulse" />
          </div>
        </div>

        {/* Cart Items List */}
        <div className="flex-1 overflow-y-auto p-8 space-y-4 no-scrollbar">
          <AnimatePresence>
            {cart.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="h-full flex flex-col items-center justify-center text-gray-600 space-y-4"
              >
                <Scan size={48} strokeWidth={1} />
                <p className="text-sm">Place items in the zone to scan</p>
              </motion.div>
            ) : (
              cart.map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: 20, scale: 0.95 }}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="group bg-white/5 hover:bg-white/10 border border-white/5 rounded-2xl p-4 flex items-center justify-between transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="h-10 w-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xs font-bold">
                      {item.name[0]}
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-200 capitalize">{item.name.replace('_', ' ')}</h3>
                      <p className="text-xs text-gray-500">Qty: {item.qty}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-mono font-bold text-lg">₹{item.price}</p>
                  </div>
                </motion.div>
              ))
            )}
            <div ref={lastItemRef} />
          </AnimatePresence>
        </div>

        {/* Footer / Total */}
        <div className="p-8 bg-black/40 border-t border-white/10 space-y-6">
          <div className="space-y-2">
            <div className="flex justify-between text-gray-400 text-sm">
              <span>Subtotal</span>
              <span>₹{total}</span>
            </div>
            <div className="flex justify-between text-gray-400 text-sm">
              <span>Tax (5%)</span>
              <span>₹{Math.round(total * 0.05)}</span>
            </div>
            <div className="flex justify-between items-end pt-4">
              <span className="text-xl font-bold">Total</span>
              <span className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-green-400">
                ₹{Math.round(total * 1.05)}
              </span>
            </div>
          </div>

          <button
            disabled={cart.length === 0}
            className="w-full bg-white text-black font-bold text-lg py-4 rounded-xl hover:bg-gray-200 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <CreditCard size={20} />
            Pay Now
          </button>
        </div>

      </div>
    </div>
  );
}

export default App;
