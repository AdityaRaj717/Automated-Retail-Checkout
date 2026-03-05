"use client";

export default function BillPanel({
    items,
    total,
    totalItems,
    onIncrement,
    onDecrement,
    onDelete,
    onClearAll,
    onCheckout,
}) {
    return (
        <aside className="bill-panel">
            {/* Header */}
            <div className="bill-header">
                <h2>Current Bill</h2>
                <div className="bill-header-meta">
                    <span className="bill-item-count">
                        {totalItems} item{totalItems !== 1 ? "s" : ""}
                    </span>
                    {items.length > 0 && (
                        <button className="bill-clear-btn" onClick={onClearAll}>
                            Clear All
                        </button>
                    )}
                </div>
            </div>

            {/* Item List */}
            <div className="bill-items">
                {items.length === 0 ? (
                    <div className="bill-empty">
                        <div className="bill-empty-icon">🛍️</div>
                        <p>No items scanned</p>
                        <p className="hint">
                            Point camera at a product and press Capture
                        </p>
                    </div>
                ) : (
                    items.map((item) => (
                        <div className="bill-item" key={item.slug}>
                            <div className="bill-item-info">
                                <div className="bill-item-name">{item.name}</div>
                                <div className="bill-item-price">₹{item.price} each</div>
                                {item.confidence && (
                                    <div className="bill-item-confidence">
                                        {(item.confidence * 100).toFixed(0)}% match
                                    </div>
                                )}
                            </div>

                            {/* Qty controls */}
                            <div className="quantity-controls">
                                <button
                                    className="qty-btn"
                                    onClick={() => onDecrement(item.slug)}
                                >
                                    −
                                </button>
                                <span className="qty-value">{item.quantity}</span>
                                <button
                                    className="qty-btn"
                                    onClick={() => onIncrement(item.slug)}
                                >
                                    +
                                </button>
                            </div>

                            {/* Subtotal */}
                            <div className="bill-item-subtotal">
                                ₹{(item.price * item.quantity).toFixed(0)}
                            </div>

                            {/* Delete */}
                            <button
                                className="bill-item-delete"
                                onClick={() => onDelete(item.slug)}
                                title="Remove item"
                            >
                                ✕
                            </button>
                        </div>
                    ))
                )}
            </div>

            {/* Footer */}
            <div className="bill-footer">
                <div className="bill-total-row">
                    <span className="bill-total-label">Total</span>
                    <span className="bill-total-amount">₹{total.toFixed(0)}</span>
                </div>
                <button
                    className="btn-checkout"
                    onClick={onCheckout}
                    disabled={items.length === 0}
                >
                    Checkout
                </button>
            </div>
        </aside>
    );
}
