"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import CameraFeed from "./components/CameraFeed";
import VisionMaps from "./components/VisionMaps";
import DetectionInfo from "./components/DetectionInfo";
import BillPanel from "./components/BillPanel";
import Toast from "./components/Toast";

const API_URL = "http://localhost:8000";

export default function Home() {
  const [billItems, setBillItems] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [cameraConnected, setCameraConnected] = useState(false);
  const [toast, setToast] = useState(null);
  const [lastDetectionCount, setLastDetectionCount] = useState(0);
  const [latestDetections, setLatestDetections] = useState([]);
  const [captureCount, setCaptureCount] = useState(0);
  const toastTimeout = useRef(null);

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

  const addToBill = useCallback((item) => {
    setBillItems((prev) => {
      const existing = prev.findIndex((b) => b.slug === item.slug);
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = { ...updated[existing], quantity: updated[existing].quantity + 1 };
        return updated;
      }
      return [...prev, { id: item.id, slug: item.slug, name: item.name, price: item.price, quantity: 1, confidence: item.confidence }];
    });
  }, []);

  // ── Capture ─────────────────────────────────────────────────────
  const handleCapture = useCallback(async () => {
    if (isCapturing) return;
    setIsCapturing(true);

    try {
      const res = await fetch(`${API_URL}/capture`, { method: "POST" });
      const data = await res.json();

      setCaptureCount((c) => c + 1);

      if (data.detections && data.detections.length > 0) {
        setLastDetectionCount(data.detections.length);
        setLatestDetections(data.detections);

        const autoAdded = [];
        for (const det of data.detections) {
          if (!det.needs_confirmation && det.product) {
            addToBill({
              id: det.product.id, slug: det.product.slug,
              name: det.product.name, price: det.product.price,
              confidence: det.confidence,
            });
            autoAdded.push(det.product.name);
          }
        }

        const needsConfirm = data.detections.filter((d) => d.needs_confirmation);
        if (autoAdded.length > 0 && needsConfirm.length === 0) {
          showToast(`Detected: ${autoAdded.join(", ")}`, "success");
        } else if (needsConfirm.length > 0) {
          showToast(`${autoAdded.length} added, ${needsConfirm.length} need confirmation`, "info");
        }
      } else {
        setLastDetectionCount(0);
        setLatestDetections([]);
        showToast("No products detected. Try adjusting position.", "warning");
      }
    } catch {
      showToast("Capture failed. Is the server running?", "warning");
    } finally {
      setTimeout(() => setIsCapturing(false), 400);
    }
  }, [isCapturing, showToast, addToBill]);

  const handleConfirm = useCallback((i, product) => {
    addToBill({ id: product.id || 0, slug: product.slug, name: product.name, price: product.price, confidence: product.confidence });
    setLatestDetections((prev) => {
      const u = [...prev];
      u[i] = { ...u[i], needs_confirmation: false, resolved: true };
      return u;
    });
    showToast(`Confirmed: ${product.name}`, "success");
  }, [addToBill, showToast]);

  const handleDismiss = useCallback((i) => {
    setLatestDetections((prev) => {
      const u = [...prev];
      u[i] = { ...u[i], needs_confirmation: false, dismissed: true };
      return u;
    });
  }, []);

  const handleResetBackground = useCallback(async () => {
    try {
      await fetch(`${API_URL}/set_background`, { method: "POST" });
      showToast("Background reference updated", "info");
    } catch { showToast("Failed to reset background", "warning"); }
  }, [showToast]);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.code === "Space" && !e.repeat) { e.preventDefault(); handleCapture(); }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleCapture]);

  const handleIncrement = (slug) => setBillItems((p) => p.map((i) => i.slug === slug ? { ...i, quantity: i.quantity + 1 } : i));
  const handleDecrement = (slug) => setBillItems((p) => p.map((i) => i.slug === slug ? { ...i, quantity: i.quantity - 1 } : i).filter((i) => i.quantity > 0));
  const handleDeleteItem = (slug) => setBillItems((p) => p.filter((i) => i.slug !== slug));
  const handleClearAll = () => { setBillItems([]); setLastDetectionCount(0); setLatestDetections([]); showToast("Bill cleared", "info"); };

  const handleCheckout = async () => {
    if (billItems.length === 0) return;
    try {
      const items = billItems.map((i) => ({ product_id: i.id, quantity: i.quantity, subtotal: i.price * i.quantity }));
      const res = await fetch(`${API_URL}/transactions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ items }) });
      if (res.ok) {
        const total = billItems.reduce((s, i) => s + i.price * i.quantity, 0);
        showToast(`Checkout complete! Total: ₹${total}`, "success");
        setBillItems([]); setLastDetectionCount(0); setLatestDetections([]);
      }
    } catch { showToast("Checkout failed", "warning"); }
  };

  const total = billItems.reduce((s, i) => s + i.price * i.quantity, 0);
  const totalItems = billItems.reduce((s, i) => s + i.quantity, 0);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1><span className="icon">🛒</span> Retail Checkout System</h1>
        <div className="header-status">
          <div className="status-badge">
            <span className={`status-dot ${cameraConnected ? "connected" : "disconnected"}`} />
            {cameraConnected ? "Camera Connected" : "Camera Offline"}
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Column 1: Camera + Controls */}
        <CameraFeed
          apiUrl={API_URL}
          isCapturing={isCapturing}
          cameraConnected={cameraConnected}
          lastDetectionCount={lastDetectionCount}
          onCapture={handleCapture}
          onResetBackground={handleResetBackground}
        />

        {/* Column 2: Vision Maps + Detection Analytics */}
        <div className="analysis-column">
          <VisionMaps apiUrl={API_URL} hasCapture={captureCount} />
          <DetectionInfo detections={latestDetections} onConfirm={handleConfirm} onDismiss={handleDismiss} />
        </div>

        {/* Column 3: Bill */}
        <BillPanel
          items={billItems} total={total} totalItems={totalItems}
          onIncrement={handleIncrement} onDecrement={handleDecrement}
          onDelete={handleDeleteItem} onClearAll={handleClearAll} onCheckout={handleCheckout}
        />
      </main>

      {toast && <Toast key={toast.key} message={toast.message} type={toast.type} />}
    </div>
  );
}
