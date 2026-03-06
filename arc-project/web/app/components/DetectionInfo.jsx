"use client";

export default function DetectionInfo({ detections, onConfirm, onDismiss }) {
    if (!detections || detections.length === 0) return null;

    const hasConfirmations = detections.some((d) => d.needs_confirmation);

    return (
        <div className="detection-info-panel">
            <div className="detection-info-header">
                <h3>🔬 Detection Analytics</h3>
                <span className="detection-info-count">
                    {detections.length} product{detections.length !== 1 ? "s" : ""}
                </span>
            </div>

            <div className="detection-info-list">
                {detections.map((det, i) => (
                    <div
                        key={i}
                        className={`detection-info-card ${det.needs_confirmation ? "needs-confirm" : ""}`}
                    >
                        {/* Product Header */}
                        <div className="detection-card-header">
                            <span className="detection-product-name">
                                {det.product?.name || det.label}
                            </span>
                            <span
                                className={`detection-confidence ${det.confidence >= 0.8
                                        ? "high"
                                        : det.confidence >= 0.6
                                            ? "medium"
                                            : "low"
                                    }`}
                            >
                                {(det.confidence * 100).toFixed(0)}%
                            </span>
                        </div>

                        {/* Price */}
                        <div className="detection-card-price">
                            ₹{det.product?.price || 0}
                            {det.product?.variant_resolved && (
                                <span className="variant-badge">SIZE VARIANT</span>
                            )}
                        </div>

                        {/* Depth Metrics */}
                        {det.depth_metrics && (
                            <div className="detection-metrics">
                                <div className="metric">
                                    <span className="metric-label">Depth</span>
                                    <span className="metric-value">
                                        {det.depth_metrics.mean_depth.toFixed(1)}
                                    </span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Volume</span>
                                    <span className="metric-value">
                                        {det.depth_metrics.estimated_volume.toFixed(0)}
                                    </span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Size</span>
                                    <span className={`metric-value size-${det.depth_metrics.size_category}`}>
                                        {det.depth_metrics.size_category}
                                    </span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Area</span>
                                    <span className="metric-value">
                                        {det.depth_metrics.bbox_area}px²
                                    </span>
                                </div>
                            </div>
                        )}

                        {/* Confirmation Panel */}
                        {det.needs_confirmation && det.candidates && (
                            <div className="confirmation-panel">
                                <div className="confirmation-label">
                                    ⚠️ Is this the correct product?
                                </div>
                                <div className="confirmation-options">
                                    {det.candidates.map((c, j) => (
                                        <button
                                            key={j}
                                            className={`confirm-btn ${j === 0 ? "primary" : ""}`}
                                            onClick={() =>
                                                onConfirm(i, {
                                                    slug: c.slug,
                                                    name: c.name,
                                                    price: c.price,
                                                    confidence: c.confidence,
                                                })
                                            }
                                        >
                                            <span className="confirm-name">{c.name}</span>
                                            <span className="confirm-conf">
                                                {(c.confidence * 100).toFixed(0)}%
                                            </span>
                                            <span className="confirm-price">₹{c.price}</span>
                                        </button>
                                    ))}
                                    <button
                                        className="confirm-btn skip"
                                        onClick={() => onDismiss(i)}
                                    >
                                        Skip
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
