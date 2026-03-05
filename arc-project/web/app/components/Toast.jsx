"use client";

import { useState, useEffect } from "react";

export default function Toast({ message, type = "info" }) {
    const [exiting, setExiting] = useState(false);

    useEffect(() => {
        const timer = setTimeout(() => setExiting(true), 2600);
        return () => clearTimeout(timer);
    }, []);

    return (
        <div className={`toast ${type} ${exiting ? "exit" : ""}`}>
            {type === "success" && "✓ "}
            {type === "warning" && "⚠ "}
            {type === "info" && "ℹ "}
            {message}
        </div>
    );
}
