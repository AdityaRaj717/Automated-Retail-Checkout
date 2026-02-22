import { useState, useEffect } from 'react'
import io from 'socket.io-client'

const socket = io('http://localhost:3000')

function App() {
  const [image, setImage] = useState(null)
  const [cart, setCart] = useState([])
  const [isScanning, setIsScanning] = useState(false)

  // Active Learning State
  const [cashierPrompt, setCashierPrompt] = useState(null)

  useEffect(() => {
    socket.on('live_feed', (data) => setImage(`data:image/jpeg;base64,${data.image}`))
    socket.on('cart_sync', (newCart) => setCart(newCart))

    // Active Learning: Cashier confirmation prompt
    socket.on('cashier_prompt_ui', (data) => {
      setCashierPrompt(data)
    })

    return () => socket.off()
  }, [])

  // Keyboard Trigger
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && !cashierPrompt) {
        e.preventDefault();
        socket.emit('request_scan');
        setIsScanning(true);
        setTimeout(() => setIsScanning(false), 2000);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [cashierPrompt]);

  // --- ACTIONS ---
  const sendAction = (type, id = null) => {
    socket.emit('cart_action', { type, id });
  };

  // Active Learning: Cashier confirms or corrects
  const handleCashierConfirm = (confirmed, correctedName = null) => {
    if (!cashierPrompt) return;
    socket.emit('cashier_response', {
      item_id: cashierPrompt.item_id,
      confirmed,
      corrected_name: correctedName
    });
    setCashierPrompt(null);
  };

  const total = cart.reduce((sum, item) => sum + item.total_price, 0);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans selection:bg-blue-500 selection:text-white">
      {/* Navbar */}
      <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold">A</div>
            <span className="text-xl font-bold tracking-tight">ARC System</span>
            <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded-full ml-2">v2.0</span>
          </div>
          <div className="flex items-center gap-4">
            <span className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${isScanning ? 'bg-green-500/20 text-green-400' : 'bg-gray-800 text-gray-400'}`}>
              {isScanning ? '● SCANNING' : '● READY'}
            </span>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-4 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Left: Camera Feed */}
        <div className="lg:col-span-7 space-y-6">
          <div className="relative rounded-2xl overflow-hidden bg-gray-800 border border-gray-700 shadow-2xl aspect-video group">
            {image ? (
              <img src={image} alt="Live Feed" className="w-full h-full object-cover" />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <p>Waiting for camera feed...</p>
              </div>
            )}

            {/* Scan Overlay */}
            <div className={`absolute inset-0 border-4 border-blue-500 transition-opacity duration-300 pointer-events-none ${isScanning ? 'opacity-100' : 'opacity-0'}`} />
          </div>

          <div className="p-6 rounded-2xl bg-gray-800/50 border border-gray-700 backdrop-blur-sm">
            <h3 className="text-lg font-semibold mb-2">Instructions</h3>
            <p className="text-gray-400 text-sm">Place items under the camera and press <kbd className="bg-gray-700 px-2 py-1 rounded text-white font-mono">Spacebar</kbd> to scan.</p>
            <div className="flex gap-4 mt-3">
              <div className="flex items-center gap-1.5 text-xs text-gray-500">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                Depth Estimation
              </div>
              <div className="flex items-center gap-1.5 text-xs text-gray-500">
                <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                Metric Learning
              </div>
              <div className="flex items-center gap-1.5 text-xs text-gray-500">
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                Active Learning
              </div>
            </div>
          </div>
        </div>

        {/* Right: Cart */}
        <div className="lg:col-span-5 flex flex-col h-[calc(100vh-8rem)] sticky top-24">
          <div className="flex-1 flex flex-col rounded-2xl bg-gray-800 border border-gray-700 shadow-xl overflow-hidden">
            <div className="p-6 border-b border-gray-700 flex justify-between items-center">
              <h2 className="text-xl font-bold">Current Cart</h2>
              <span className="text-sm text-gray-400">{cart.length} Items</span>
            </div>

            {/* Scrollable List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {cart.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-gray-500 opacity-50">
                  <p>Cart is empty</p>
                </div>
              ) : (
                cart.map((item) => (
                  <div key={item.id} className="flex items-center justify-between p-4 rounded-xl bg-gray-700/30 border border-gray-700/50 hover:border-blue-500/30 transition-all group">
                    <div>
                      <h4 className="font-semibold text-lg">{item.display_name}</h4>
                      <p className="text-sm text-gray-400">₹{item.unit_price} x {item.qty}</p>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="flex items-center bg-gray-800 rounded-lg border border-gray-600">
                        <button onClick={() => sendAction('decrement', item.id)} className="px-3 py-1 hover:bg-gray-700 text-blue-400 font-bold">-</button>
                        <span className="px-2 font-mono text-sm">{item.qty}</span>
                        <button onClick={() => sendAction('increment', item.id)} className="px-3 py-1 hover:bg-gray-700 text-blue-400 font-bold">+</button>
                      </div>
                      <div className="text-right min-w-[60px]">
                        <p className="font-bold text-green-400">₹{item.total_price}</p>
                      </div>
                      <button onClick={() => sendAction('remove', item.id)} className="p-2 text-red-400 hover:bg-red-500/10 rounded-lg transition-colors">
                        ✕
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Footer Total */}
            <div className="p-6 bg-gray-800 border-t border-gray-700">
              <div className="flex justify-between items-end mb-4">
                <span className="text-gray-400">Total Amount</span>
                <span className="text-3xl font-bold text-white">₹{total}</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <button onClick={() => sendAction('clear')} className="w-full py-3 rounded-xl font-semibold bg-gray-700 text-gray-300 hover:bg-gray-600 transition-all">
                  Clear Cart
                </button>
                <button className="w-full py-3 rounded-xl font-bold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20 transition-all">
                  Checkout
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* === ACTIVE LEARNING: Cashier Confirmation Modal === */}
      {cashierPrompt && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
          <div className="bg-gray-800 rounded-2xl border border-gray-600 shadow-2xl max-w-md w-full overflow-hidden animate-in">
            {/* Header */}
            <div className="px-6 pt-6 pb-4">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-3 h-3 rounded-full bg-amber-500 animate-pulse"></span>
                <h3 className="text-lg font-bold text-amber-400">Confirmation Required</h3>
              </div>
              <p className="text-sm text-gray-400">The AI is unsure about this item. Please verify.</p>
            </div>

            {/* Product Image */}
            <div className="px-6 pb-4">
              <div className="rounded-xl overflow-hidden bg-gray-900 border border-gray-700 aspect-square max-w-[200px] mx-auto">
                <img
                  src={`data:image/jpeg;base64,${cashierPrompt.image_b64}`}
                  alt="Detected item"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            {/* Info */}
            <div className="px-6 pb-4 text-center">
              <p className="text-2xl font-bold text-white mb-1">{cashierPrompt.suggested_name}</p>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30">
                <span className="text-sm text-amber-400">Confidence: {cashierPrompt.confidence}%</span>
              </div>
            </div>

            {/* Actions */}
            <div className="p-6 bg-gray-900/50 border-t border-gray-700 space-y-3">
              <p className="text-sm text-gray-400 text-center mb-3">Is this <strong className="text-white">{cashierPrompt.suggested_name}</strong>?</p>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => handleCashierConfirm(false)}
                  className="w-full py-3 rounded-xl font-semibold bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 transition-all"
                >
                  ✕ No, Wrong
                </button>
                <button
                  onClick={() => handleCashierConfirm(true)}
                  className="w-full py-3 rounded-xl font-bold bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-500/20 transition-all"
                >
                  ✓ Yes, Correct
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
