"use client";

import { useState, useEffect } from "react";

export default function VisionMaps({ apiUrl, hasCapture }) {
    const [timestamp, setTimestamp] = useState(0);

    // Refresh maps when a new capture happens
    useEffect(() => {
        if (hasCapture) {
            setTimestamp(Date.now());
        }
    }, [hasCapture]);

    return (
        <div className="vision-maps">
            <div className="vision-map-card">
                <div className="vision-map-header">
                    <span className="vision-map-icon">🗺️</span>
                    <span className="vision-map-title">Depth Map</span>
                    <span className="vision-map-badge depth">DAv2</span>
                </div>
                <div className="vision-map-image">
                    {timestamp > 0 ? (
                        <img
                            src={`${apiUrl}/depth_map?t=${timestamp}`}
                            alt="Depth map"
                            draggable={false}
                            onError={(e) => {
                                e.target.style.display = "none";
                                e.target.nextSibling.style.display = "flex";
                            }}
                        />
                    ) : null}
                    <div className="vision-map-placeholder" style={timestamp > 0 ? { display: "none" } : {}}>
                        <span>Press Capture to generate</span>
                    </div>
                </div>
            </div>

            <div className="vision-map-card">
                <div className="vision-map-header">
                    <span className="vision-map-icon">🔮</span>
                    <span className="vision-map-title">Ambient Occlusion</span>
                    <span className="vision-map-badge occlusion">SSAO</span>
                </div>
                <div className="vision-map-image">
                    {timestamp > 0 ? (
                        <img
                            src={`${apiUrl}/occlusion_map?t=${timestamp}`}
                            alt="Ambient occlusion"
                            draggable={false}
                            onError={(e) => {
                                e.target.style.display = "none";
                                e.target.nextSibling.style.display = "flex";
                            }}
                        />
                    ) : null}
                    <div className="vision-map-placeholder" style={timestamp > 0 ? { display: "none" } : {}}>
                        <span>Press Capture to generate</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
