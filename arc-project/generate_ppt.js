const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Capstone Team';
pres.title = 'Intelligent Retail Checkout System';

// Theme Colors: Midnight Executive + Teal Accent
const C_NAVY = "1E2761";
const C_ICE = "CADCFC";
const C_WHITE = "FFFFFF";
const C_TEAL = "00A896";
const C_GRAY = "E2E8F0";
const C_DARKGRAY = "333333";

// Define Master Slides
pres.defineSlideMaster({
    title: 'TITLE_SLIDE',
    background: { color: C_NAVY },
    objects: [
        { rect: { x: 0, y: 4.5, w: '100%', h: 1.125, fill: { color: C_TEAL } } }
    ]
});

pres.defineSlideMaster({
    title: 'DARK_CONTENT',
    background: { color: C_NAVY },
    objects: [
        { rect: { x: 0.5, y: 0.8, w: 9, h: 0.05, fill: { color: C_TEAL } } },
        { text: { text: "Intelligent Retail Checkout System", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: C_ICE, fontFace: "Calibri" } } }
    ]
});

pres.defineSlideMaster({
    title: 'LIGHT_CONTENT',
    background: { color: C_WHITE },
    objects: [
        { rect: { x: 0.5, y: 0.8, w: 9, h: 0.05, fill: { color: C_TEAL } } },
        { text: { text: "Intelligent Retail Checkout System", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: "64748B", fontFace: "Calibri" } } }
    ]
});

// --- Slide 1: Title ---
let slide1 = pres.addSlide({ masterName: "TITLE_SLIDE" });
slide1.addText("Intelligent Retail", { x: 0.5, y: 1.5, w: 8, h: 0.8, fontSize: 48, color: C_WHITE, bold: true, fontFace: "Arial Black", margin: 0 });
slide1.addText("Checkout System", { x: 0.5, y: 2.3, w: 9, h: 0.8, fontSize: 44, color: C_TEAL, bold: true, fontFace: "Arial Black", margin: 0 });
slide1.addText("Powered by Custom Vision AI Architecture", { x: 0.5, y: 3.2, w: 8, h: 0.5, fontSize: 24, color: C_ICE, fontFace: "Georgia", italic: true, margin: 0 });
slide1.addText("Capstone Project Presentation", { x: 0.5, y: 4.8, w: 8, h: 0.5, fontSize: 20, color: C_WHITE, fontFace: "Calibri", bold: true, margin: 0 });

// --- Slide 2: Problem Statement ---
let slide2 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide2.addText("The Problem", { x: 0.5, y: 0.2, w: 8, h: 0.6, fontSize: 36, color: C_NAVY, bold: true, fontFace: "Arial Black" });

slide2.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 4.2, h: 3.5, fill: { color: "F8FAFC" }, line: { color: C_GRAY, width: 1 } });
slide2.addText("Retail Inefficiencies", { x: 0.7, y: 1.4, w: 3.8, h: 0.4, fontSize: 20, color: C_NAVY, bold: true, fontFace: "Arial" });
slide2.addText([
    { text: "Manual checkout is slow and prone to human error.", options: { bullet: true, breakLine: true } },
    { text: "Traditional barcoding struggles with small, loose, or unstructured items.", options: { bullet: true, breakLine: true } },
    { text: "Adding new products requires system-wide updates.", options: { bullet: true } }
], { x: 0.7, y: 2.0, w: 3.8, h: 2, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri", valign: "top" });

slide2.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 1.2, w: 4.2, h: 3.5, fill: { color: "F8FAFC" }, line: { color: C_GRAY, width: 1 } });
slide2.addText("Limitations of Base Models", { x: 5.5, y: 1.4, w: 3.8, h: 0.4, fontSize: 20, color: C_NAVY, bold: true, fontFace: "Arial" });
slide2.addText([
    { text: "Off-the-shelf single-object segmentation models fail on clustered products.", options: { bullet: true, breakLine: true } },
    { text: "Aggressive filtering destroys small items (e.g. matchboxes).", options: { bullet: true, breakLine: true } },
    { text: "Cannot distinguish size variants of visually identical packaging.", options: { bullet: true } }
], { x: 5.5, y: 2.0, w: 3.8, h: 2, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri", valign: "top" });


// --- Slide 3: Our Approach ---
let slide3 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide3.addText("Our Approach: Custom AI Pipeline", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_WHITE, bold: true, fontFace: "Arial Black" });

let blocks = [
    { title: "1. Hybrid Detection", desc: "Dichotomous Segmentation + Sliding Window to capture every item.", x: 0.5, color: "028090" },
    { title: "2. Metric Learning", desc: "Deep Feature Extractor mapping items to embeddings for kNN hot-reload.", x: 3.66, color: "00A896" },
    { title: "3. Depth Analysis", desc: "Monocular Depth Estimation to resolve physical size and volume.", x: 6.83, color: "02C39A" }
];

blocks.forEach(b => {
    slide3.addShape(pres.shapes.RECTANGLE, { x: b.x, y: 1.8, w: 2.66, h: 0.5, fill: { color: b.color } });
    slide3.addText(b.title, { x: b.x, y: 1.8, w: 2.66, h: 0.5, fontSize: 16, color: C_WHITE, bold: true, align: "center", fontFace: "Arial" });

    slide3.addShape(pres.shapes.RECTANGLE, { x: b.x, y: 2.3, w: 2.66, h: 1.8, fill: { color: "111827" }, line: { color: "374151", width: 1 } });
    slide3.addText(b.desc, { x: b.x + 0.1, y: 2.4, w: 2.46, h: 1.6, fontSize: 14, color: "E5E7EB", fontFace: "Calibri", align: "center", valign: "middle" });
});

// Arrow shapes
slide3.addShape(pres.shapes.RIGHT_ARROW, { x: 3.25, y: 2.0, w: 0.3, h: 0.2, fill: { color: C_ICE } });
slide3.addShape(pres.shapes.RIGHT_ARROW, { x: 6.42, y: 2.0, w: 0.3, h: 0.2, fill: { color: C_ICE } });


// --- Slide 4: Stage 1 ---
let slide4 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide4.addText("Stage 1: Hybrid Detection Engine", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_NAVY, bold: true, fontFace: "Arial Black" });

slide4.addText("Problem:", { x: 0.5, y: 1.1, w: 9, h: 0.4, fontSize: 18, color: "EF4444", bold: true, fontFace: "Arial" });
slide4.addText("Standard salient object segmentation aggressively erodes small items like tic-tacs.", { x: 0.5, y: 1.5, w: 9, h: 0.4, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri" });

slide4.addText("Our Custom Solution:", { x: 0.5, y: 2.1, w: 9, h: 0.4, fontSize: 18, color: C_TEAL, bold: true, fontFace: "Arial" });
slide4.addText([
    { text: "Custom Retail Segmenter: Optimized for multiple, tightly clustered products.", options: { bullet: true, breakLine: true } },
    { text: "Sliding Window Scanner: A sophisticated fallback mechanism that scans the raw frame for unmasked blobs to ensure 100% recall of small items.", options: { bullet: true } }
], { x: 0.5, y: 2.5, w: 9, h: 1.0, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri" });

slide4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.0, w: 9, h: 0.8, fill: { color: "F1F5F9" }, line: { color: "CBD5E1", width: 1 } });
slide4.addText("Academic Foundation / Citation:", { x: 0.6, y: 4.1, w: 8.8, h: 0.3, fontSize: 12, color: C_NAVY, bold: true, fontFace: "Arial" });
slide4.addText("Inspired by: Qin, X., et al. (2022). \"Highly Accurate Dichotomous Image Segmentation\". ECCV.", { x: 0.6, y: 4.4, w: 8.8, h: 0.3, fontSize: 12, color: "475569", fontFace: "Calibri", italic: true });


// --- Slide 5: Stage 2 ---
let slide5 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide5.addText("Stage 2: Deep Metric Learning Classifier", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_NAVY, bold: true, fontFace: "Arial Black" });

slide5.addText("Problem:", { x: 0.5, y: 1.1, w: 9, h: 0.4, fontSize: 18, color: "EF4444", bold: true, fontFace: "Arial" });
slide5.addText("Retraining an entire categorization network for every new retail product is unscalable.", { x: 0.5, y: 1.5, w: 9, h: 0.4, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri" });

slide5.addText("Our Custom Solution:", { x: 0.5, y: 2.1, w: 9, h: 0.4, fontSize: 18, color: C_TEAL, bold: true, fontFace: "Arial" });
slide5.addText([
    { text: "Deep Feature Extractor: Maps visual features into a high-dimensional embedding space.", options: { bullet: true, breakLine: true } },
    { text: "k-Nearest Neighbors (kNN) Engine: Classifies items by comparing embeddings.", options: { bullet: true, breakLine: true } },
    { text: "Hot-Reloading: Adding new products only requires saving images to a folder and generating embeddings dynamically—zero model retraining required.", options: { bullet: true } }
], { x: 0.5, y: 2.5, w: 9, h: 1.2, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri" });

slide5.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.1, w: 9, h: 0.8, fill: { color: "F1F5F9" }, line: { color: "CBD5E1", width: 1 } });
slide5.addText("Academic Foundation / Citation:", { x: 0.6, y: 4.2, w: 8.8, h: 0.3, fontSize: 12, color: C_NAVY, bold: true, fontFace: "Arial" });
slide5.addText("Derived from: Schroff, F., et al. (2015). \"FaceNet: A Unified Embedding for Face Recognition and Clustering\". CVPR.", { x: 0.6, y: 4.5, w: 8.8, h: 0.3, fontSize: 12, color: "475569", fontFace: "Calibri", italic: true });


// --- Slide 6: Stage 3 ---
let slide6 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide6.addText("Stage 3: Monocular Depth Analysis", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_NAVY, bold: true, fontFace: "Arial Black" });

slide6.addText("Problem:", { x: 0.5, y: 1.1, w: 9, h: 0.4, fontSize: 18, color: "EF4444", bold: true, fontFace: "Arial" });
slide6.addText("Visually identical packaging with different sizes (e.g., 10rs vs 30rs packets) are misclassified.", { x: 0.5, y: 1.5, w: 9, h: 0.4, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri" });

slide6.addText("Our Custom Solution:", { x: 0.5, y: 2.1, w: 9, h: 0.4, fontSize: 18, color: C_TEAL, bold: true, fontFace: "Arial" });
slide6.addText([
    { text: "Integrated a Monocular Depth Analyzer to generate real-time relative depth maps.", options: { bullet: true, breakLine: true } },
    { text: "Calculates metric depth, physical size, and estimated volume per bounding box.", options: { bullet: true, breakLine: true } },
    { text: "Variant Resolver automatically categorizes the correct product tier based on volume thresholds.", options: { bullet: true } }
], { x: 0.5, y: 2.5, w: 9, h: 1.2, fontSize: 16, color: C_DARKGRAY, fontFace: "Calibri" });

slide6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.1, w: 9, h: 0.8, fill: { color: "F1F5F9" }, line: { color: "CBD5E1", width: 1 } });
slide6.addText("Academic Foundation / Citation:", { x: 0.6, y: 4.2, w: 8.8, h: 0.3, fontSize: 12, color: C_NAVY, bold: true, fontFace: "Arial" });
slide6.addText("Methodology aligned with: Yang, L., et al. (2024). \"Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data\". CVPR.", { x: 0.6, y: 4.5, w: 8.8, h: 0.3, fontSize: 12, color: "475569", fontFace: "Calibri", italic: true });


// --- Slide 7: Ambiguity ---
let slide7 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide7.addText("Human-in-the-Loop: Ambiguity Resolution", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_WHITE, bold: true, fontFace: "Arial Black" });

slide7.addText([
    { text: "Confidence Thresholding:", options: { bold: true, breakLine: true } },
    { text: "If the classifier confidence is marginal (0.45 - 0.70) or the delta between the top 2 candidates is < 0.15, the system flags the detection as ambiguous.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "UI Integration:", options: { bold: true, breakLine: true } },
    { text: "The Next.js dashboard halts auto-checkout for flagged items, presenting the cashier with the top 3 visually similar candidates with pricing.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Result:", options: { bold: true, breakLine: true } },
    { text: "Ensures 100% checkout accuracy while maintaining high throughput for clear detections.", options: {} }
], { x: 0.5, y: 1.3, w: 9, h: 3, fontSize: 18, color: C_ICE, fontFace: "Calibri", valign: "top" });


// --- Slide 8: Datasets ---
let slide8 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide8.addText("Custom Dataset Engineering", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_NAVY, bold: true, fontFace: "Arial Black" });

slide8.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 9, h: 1.2, fill: { color: C_NAVY }, rectRadius: 0.1 });
slide8.addText("We collected, curated, and annotated a retail dataset completely from scratch.", { x: 0.5, y: 1.3, w: 9, h: 1.0, fontSize: 22, color: C_WHITE, bold: true, fontFace: "Arial", align: "center", margin: 0.2 });

slide8.addText([
    { text: "Varied Lighting:", options: { bold: true } },
    { text: " Captured under extreme shadows, glare, and low-light.", options: { breakLine: true } },
    { text: "Cluster Densities:", options: { bold: true } },
    { text: " Piled, stacked, and touching products.", options: { breakLine: true } },
    { text: "Occlusions:", options: { bold: true } },
    { text: " Products partially obscuring each other to train the segmentation models robustly.", options: { breakLine: true } }
], { x: 0.5, y: 2.8, w: 9, h: 2.0, fontSize: 18, color: C_DARKGRAY, fontFace: "Calibri", valign: "top" });


// --- Slide 9: Architecture Recap ---
let slide9 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide9.addText("System Architecture Integration", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_WHITE, bold: true, fontFace: "Arial Black" });

slide9.addShape(pres.shapes.RECTANGLE, { x: 1, y: 1.5, w: 2.3, h: 2, fill: { color: "1F2937" }, line: { color: C_TEAL, width: 2 } });
slide9.addText("Hardware", { x: 1, y: 1.6, w: 2.3, h: 0.4, fontSize: 16, color: C_TEAL, bold: true, align: "center" });
slide9.addText("DroidCam Video Feed\nhigh-res capture", { x: 1, y: 2.2, w: 2.3, h: 1, fontSize: 14, color: C_ICE, align: "center" });

slide9.addShape(pres.shapes.RECTANGLE, { x: 3.8, y: 1.5, w: 2.3, h: 2, fill: { color: "1F2937" }, line: { color: C_TEAL, width: 2 } });
slide9.addText("Backend (FastAPI)", { x: 3.8, y: 1.6, w: 2.3, h: 0.4, fontSize: 16, color: C_TEAL, bold: true, align: "center" });
slide9.addText("Custom CV Pipeline\nSQLite Database\nHot-Reloading", { x: 3.8, y: 2.2, w: 2.3, h: 1, fontSize: 14, color: C_ICE, align: "center" });

slide9.addShape(pres.shapes.RECTANGLE, { x: 6.6, y: 1.5, w: 2.3, h: 2, fill: { color: "1F2937" }, line: { color: C_TEAL, width: 2 } });
slide9.addText("Frontend (Next.js)", { x: 6.6, y: 1.6, w: 2.3, h: 0.4, fontSize: 16, color: C_TEAL, bold: true, align: "center" });
slide9.addText("3-column Dashboard\nVision Maps (SSAO)\nCart & UX Flow", { x: 6.6, y: 2.2, w: 2.3, h: 1, fontSize: 14, color: C_ICE, align: "center" });

slide9.addShape(pres.shapes.RIGHT_ARROW, { x: 3.4, y: 2.3, w: 0.3, h: 0.2, fill: { color: C_ICE } });
slide9.addShape(pres.shapes.RIGHT_ARROW, { x: 6.2, y: 2.3, w: 0.3, h: 0.2, fill: { color: C_ICE } });


// --- Slide 10: Conclusion ---
let slide10 = pres.addSlide({ masterName: "TITLE_SLIDE" });
slide10.addText("Conclusion", { x: 0.5, y: 1.5, w: 9, h: 0.8, fontSize: 44, color: C_TEAL, bold: true, fontFace: "Arial Black", margin: 0 });
slide10.addText("A scalable, completely custom retail ecosystem.", { x: 0.5, y: 2.5, w: 9, h: 0.5, fontSize: 24, color: C_ICE, fontFace: "Georgia", italic: true, margin: 0 });
slide10.addText("Thank You.", { x: 0.5, y: 3.5, w: 9, h: 0.5, fontSize: 28, color: C_WHITE, fontFace: "Calibri", bold: true, margin: 0 });


pres.writeFile({ fileName: "Retail_Checkout_Capstone.pptx" }).then(() => {
    console.log("PPTX created successfully!");
}).catch(err => {
    console.error("Error creating PPTX", err);
});
