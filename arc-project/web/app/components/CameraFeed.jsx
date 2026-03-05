"use client";

import { useState } from "react";

export default function CameraFeed({
    apiUrl,
    isCapturing,
    cameraConnected,
    lastDetectionCount,
    onCapture,
    onResetBackground,
}) {
    const [showFlash, setShowFlash] = useState(false);

    const handleCapture = () => {
        setShowFlash(true);
        setTimeout(() => setShowFlash(false), 400);
        onCapture();
    };

    return (
        <section className="camera-panel">
            <div className="camera-feed-wrapper">
                {cameraConnected ? (
                    <>
                        <img
                            src={`${apiUrl}/video_feed`}
                            alt="Live camera feed"
                            draggable={false}
                        />

                        {/* Overlay badges */}
                        <div className="camera-overlay">
                            <span className="camera-badge live">● LIVE</span>
                            <span className="camera-badge">DroidCam</span>
                        </div>

                        {/* Detection count */}
                        {lastDetectionCount > 0 && (
                            <div className="detection-count-badge">
                                {lastDetectionCount} item{lastDetectionCount !== 1 ? "s" : ""}{" "}
                                detected
                            </div>
                        )}

                        {/* White flash on capture */}
                        {showFlash && <div className="capture-flash" />}
                    </>
                ) : (
                    <div className="camera-no-feed">
                        <div className="camera-no-feed-icon">📷</div>
                        <h3>No Camera Feed</h3>
                        <p>
                            Start the backend server and connect DroidCam
                            <br />
                            <code>python server.py</code>
                        </p>
                    </div>
                )}
            </div>

            {/* Controls */}
            <div className="camera-controls">
                <button
                    className={`btn btn-capture ${isCapturing ? "capturing" : ""}`}
                    onClick={handleCapture}
                    disabled={!cameraConnected || isCapturing}
                >
                    <span className="btn-icon">📸</span>
                    {isCapturing ? "Detecting..." : "Capture"}
                    <span className="shortcut-hint">[SPACE]</span>
                </button>
                <button
                    className="btn btn-bg-reset"
                    onClick={onResetBackground}
                    disabled={!cameraConnected}
                    title="Reset background reference"
                >
                    <span className="btn-icon">🔄</span>
                    Reset BG
                </button>
            </div>
        </section>
    );
}
