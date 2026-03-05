"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import CameraFeed from "./components/CameraFeed";
import BillPanel from "./components/BillPanel";
import Toast from "./components/Toast";

const API_URL = "http://localhost:8000";

export default function Home() {
  const [billItems, setBillItems] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [cameraConnected, setCameraConnected] = useState(false);
  const [toast, setToast] = useState(null);
  const [lastDetectionCount, setLastDetectionCount] = useState(0);
  const toastTimeout = useRef(null);

  // Check camera connection
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        setCameraConnected(data.camera_connected);
      } catch {
        setCameraConnected(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const showToast = useCallback((message, type = "info") => {
    if (toastTimeout.current) clearTimeout(toastTimeout.current);
    setToast({ message, type, key: Date.now() });
    toastTimeout.current = setTimeout(() => setToast(null), 3000);
  }, []);

  // ── Capture ─────────────────────────────────────────────────────
  const handleCapture = useCallback(async () => {
    if (isCapturing) return;
    setIsCapturing(true);

    try {
      const res = await fetch(`${API_URL}/capture`, { method: "POST" });
      const data = await res.json();

      if (data.detections && data.detections.length > 0) {
        setLastDetectionCount(data.detections.length);

        setBillItems((prev) => {
          const updated = [...prev];
          for (const det of data.detections) {
            const existing = updated.findIndex(
              (item) => item.slug === det.product.slug
            );
            if (existing >= 0) {
              // Increment quantity
              updated[existing] = {
                ...updated[existing],
                quantity: updated[existing].quantity + 1,
              };
            } else {
              // Add new item
              updated.push({
                id: det.product.id,
                slug: det.product.slug,
                name: det.product.name,
                price: det.product.price,
                quantity: 1,
                confidence: det.confidence,
              });
            }
          }
          return updated;
        });

        const names = data.detections.map((d) => d.product.name).join(", ");
        showToast(`Detected: ${names}`, "success");
      } else {
        setLastDetectionCount(0);
        showToast("No products detected. Try adjusting position.", "warning");
      }
    } catch (err) {
      showToast("Capture failed. Is the server running?", "warning");
    } finally {
      setTimeout(() => setIsCapturing(false), 400);
    }
  }, [isCapturing, showToast]);

  // ── Background Reset ────────────────────────────────────────────
  const handleResetBackground = useCallback(async () => {
    try {
      await fetch(`${API_URL}/set_background`, { method: "POST" });
      showToast("Background reference updated", "info");
    } catch {
      showToast("Failed to reset background", "warning");
    }
  }, [showToast]);

  // ── Keyboard Shortcut ───────────────────────────────────────────
  useEffect(() => {
    const handleKey = (e) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        handleCapture();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleCapture]);

  // ── Bill Operations ─────────────────────────────────────────────
  const handleIncrement = (slug) => {
    setBillItems((prev) =>
      prev.map((item) =>
        item.slug === slug ? { ...item, quantity: item.quantity + 1 } : item
      )
    );
  };

  const handleDecrement = (slug) => {
    setBillItems((prev) =>
      prev
        .map((item) =>
          item.slug === slug ? { ...item, quantity: item.quantity - 1 } : item
        )
        .filter((item) => item.quantity > 0)
    );
  };

  const handleDeleteItem = (slug) => {
    setBillItems((prev) => prev.filter((item) => item.slug !== slug));
  };

  const handleClearAll = () => {
    setBillItems([]);
    setLastDetectionCount(0);
    showToast("Bill cleared", "info");
  };

  // ── Checkout ────────────────────────────────────────────────────
  const handleCheckout = async () => {
    if (billItems.length === 0) return;

    try {
      const items = billItems.map((item) => ({
        product_id: item.id,
        quantity: item.quantity,
        subtotal: item.price * item.quantity,
      }));

      const res = await fetch(`${API_URL}/transactions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      });

      if (res.ok) {
        const total = billItems.reduce(
          (sum, item) => sum + item.price * item.quantity,
          0
        );
        showToast(`Checkout complete! Total: ₹${total}`, "success");
        setBillItems([]);
        setLastDetectionCount(0);
      }
    } catch {
      showToast("Checkout failed", "warning");
    }
  };

  const total = billItems.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );
  const totalItems = billItems.reduce((sum, item) => sum + item.quantity, 0);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>
          <span className="icon">🛒</span>
          Retail Checkout System
        </h1>
        <div className="header-status">
          <div className="status-badge">
            <span
              className={`status-dot ${cameraConnected ? "connected" : "disconnected"
                }`}
            />
            {cameraConnected ? "Camera Connected" : "Camera Offline"}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <CameraFeed
          apiUrl={API_URL}
          isCapturing={isCapturing}
          cameraConnected={cameraConnected}
          lastDetectionCount={lastDetectionCount}
          onCapture={handleCapture}
          onResetBackground={handleResetBackground}
        />
        <BillPanel
          items={billItems}
          total={total}
          totalItems={totalItems}
          onIncrement={handleIncrement}
          onDecrement={handleDecrement}
          onDelete={handleDeleteItem}
          onClearAll={handleClearAll}
          onCheckout={handleCheckout}
        />
      </main>

      {/* Toast */}
      {toast && (
        <Toast
          key={toast.key}
          message={toast.message}
          type={toast.type}
        />
      )}
    </div>
  );
}
