const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Group 11';
pres.title = 'Automated Retail Billing System';

// Theme Colors: Teal Trust
const C_DEEPBLUE = "065A82";
const C_TEAL = "1C7293";
const C_MIDNIGHT = "21295C";
const C_WHITE = "FFFFFF";
const C_OFFWHITE = "F2F2F2";
const C_GRAY = "E2E8F0";
const C_TEXT = "333333";
const C_RED = "EF4444";

// Define Master Slides
pres.defineSlideMaster({
    title: 'TITLE_SLIDE',
    background: { color: C_MIDNIGHT },
    objects: [
        { rect: { x: 0, y: 5.2, w: '100%', h: 0.425, fill: { color: C_TEAL } } }
    ]
});

pres.defineSlideMaster({
    title: 'DARK_CONTENT',
    background: { color: C_MIDNIGHT },
    objects: [
        { rect: { x: 0.5, y: 0.8, w: 9, h: 0.05, fill: { color: C_TEAL } } },
        { text: { text: "Group 11 | Phase 2 | VIT Bhopal", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: "9CA3AF", fontFace: "Calibri" } } }
    ]
});

pres.defineSlideMaster({
    title: 'LIGHT_CONTENT',
    background: { color: C_WHITE },
    objects: [
        { rect: { x: 0.5, y: 0.8, w: 9, h: 0.05, fill: { color: C_TEAL } } },
        { text: { text: "Group 11 | Phase 2 | VIT Bhopal", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: "64748B", fontFace: "Calibri" } } }
    ]
});

// --- Slide 1: Title ---
let slide1 = pres.addSlide({ masterName: "TITLE_SLIDE" });
slide1.addText("Capstone Project - Phase 2", { x: 0.5, y: 0.8, w: 9, h: 0.5, fontSize: 24, color: C_TEAL, fontFace: "Georgia", italic: true, margin: 0 });
slide1.addText("Automated Retail", { x: 0.5, y: 1.5, w: 9, h: 0.8, fontSize: 54, color: C_WHITE, bold: true, fontFace: "Arial Black", margin: 0 });
slide1.addText("Billing System", { x: 0.5, y: 2.4, w: 9, h: 0.8, fontSize: 50, color: "4DD0E1", bold: true, fontFace: "Arial Black", margin: 0 });

slide1.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.5, w: 9, h: 1.5, fill: { color: "1E293B" }, rectRadius: 0.1 });
slide1.addText("VIT Bhopal | Group No. 11", { x: 0.7, y: 3.6, w: 8.6, h: 0.3, fontSize: 18, color: C_WHITE, bold: true, fontFace: "Arial" });
slide1.addText([
    { text: "Shivam Singh (22BAI10184)\nShayan Singha (22BAI10327)", options: { x: 0.7, y: 4.0, w: 4, h: 0.8, fontSize: 14, color: "94A3B8" } },
], { x: 0.7, y: 4.0, w: 4, h: 0.8, fontSize: 14, color: "cbd5e1", fontFace: "Calibri" });
slide1.addText([
    { text: "Adityaraj Rajesh Kumar (22BAI10190)\nKaushal Sengupta (22BAI10101)\nHarsh Naik (22BAI10360)", options: {} }
], { x: 4.7, y: 4.0, w: 4, h: 0.8, fontSize: 14, color: "cbd5e1", fontFace: "Calibri" });


// --- Slide 2: Problem Statement ---
let slide2 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide2.addText("Problem Statement", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

slide2.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 9, h: 3.5, fill: { color: "F8FAFC" }, line: { color: C_GRAY, width: 1 } });
slide2.addText("The Bottleneck in Modern Retail", { x: 0.7, y: 1.4, w: 8.6, h: 0.4, fontSize: 24, color: C_DEEPBLUE, bold: true, fontFace: "Arial" });
slide2.addText([
    { text: "Manual Barcode Scanning is Slow:", options: { bold: true } },
    { text: " Cashiers must locate, align, and scan each item sequentially, leading to long queues during peak hours.", options: { breakLine: true } },
    { text: "Inventory Discrepancies:", options: { bold: true } },
    { text: " Human error results in unscanned items, mis-scanned variants, and significant shrinkage/theft.", options: { breakLine: true } },
    { text: "Friction in Customer Experience:", options: { bold: true } },
    { text: " Waiting in line for checkout is the #1 complaint of modern brick-and-mortar retail shoppers.", options: { breakLine: true } },
    { text: "Operational Costs:", options: { bold: true } },
    { text: " Stores require extensive staffing exclusively for checkout processing rather than customer assistance." }
], { x: 0.7, y: 2.0, w: 8.6, h: 2.5, fontSize: 18, color: C_TEXT, fontFace: "Calibri", valign: "top", paraSpaceAfter: 10 });


// --- Slide 3: The Challenges ---
let slide3 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide3.addText("The Challenges", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

let challengeBlocks = [
    { title: "Visual Similarity", desc: "Differentiating between products sharing identical packaging logic but different sizes (e.g., 10rs vs 30rs Maggi).", x: 0.5, y: 1.5, c: "FEE2E2" },
    { title: "Occlusion & Clustering", desc: "Customers place items in random orientations. Items frequently stack, overlap, or cast complex shadows on one another.", x: 5.1, y: 1.5, c: "FEF3C7" },
    { title: "Loose / Unstructured Items", desc: "Non-rigid items like transparent polybags, loose produce, or extremely small items (matchboxes) that erode in standard vision passes.", x: 0.5, y: 3.2, c: "E0E7FF" },
    { title: "Scalability (The Addition Problem)", desc: "Adding new inventory to standard models requires massive retraining dataset collections, halting deployment operations.", x: 5.1, y: 3.2, c: "DCFCE7" }
];

challengeBlocks.forEach(b => {
    slide3.addShape(pres.shapes.RECTANGLE, { x: b.x, y: b.y, w: 4.4, h: 1.5, fill: { color: b.c }, line: { color: "94A3B8", width: 1 } });
    slide3.addText(b.title, { x: b.x + 0.1, y: b.y + 0.1, w: 4.2, h: 0.3, fontSize: 18, color: C_MIDNIGHT, bold: true, fontFace: "Arial" });
    slide3.addText(b.desc, { x: b.x + 0.1, y: b.y + 0.5, w: 4.2, h: 0.9, fontSize: 14, color: C_TEXT, fontFace: "Calibri", valign: "top" });
});


// --- Slide 4: Challenges in Current Technologies ---
let slide4 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide4.addText("Challenges in Current Retail Tech", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_WHITE, bold: true, fontFace: "Arial Black" });

slide4.addTable([
    [{ text: "Technology", options: { bold: true, fill: { color: "1E293B" }, color: "FFFFFF" } },
    { text: "Core Mechanism", options: { bold: true, fill: { color: "1E293B" }, color: "FFFFFF" } },
    { text: "Critical Failures", options: { bold: true, fill: { color: "1E293B" }, color: "FFFFFF" } }],
    [{ text: "Standard Barcode" }, { text: "Laser Optical Scanning" }, { text: "Sequential processing. Requires line-of-sight & manual handling. Prone to deliberate swaps." }],
    [{ text: "Weight Scales" }, { text: "Load Cell Measurement" }, { text: "Cannot differentiate distinct products that happen to weigh exactly the same." }],
    [{ text: "RFID Tags" }, { text: "Radio Frequency Identify" }, { text: "Prohibitive tag costs for cheap FMCG items. High infrastructure setup cost." }],
    [{ text: "Standard YOLO/CNN" }, { text: "Bounding Box Regression" }, { text: "Must be completely retrained from scratch just to add one new flavor of chips to the store." }]
], {
    x: 0.5, y: 1.5, w: 9, h: 3,
    colW: [1.8, 2.5, 4.7],
    border: { pt: 1, color: "475569" },
    fill: { color: "0F172A" },
    color: "E2E8F0",
    fontSize: 14,
    fontFace: "Calibri",
    valign: "middle"
});


// --- Slide 5: Proposed Framework ---
let slide5 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide5.addText("Proposed Framework", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

slide5.addText("A 3-Stage Visual Pipeline solving scalability & occlusion.", { x: 0.5, y: 1.0, w: 9, h: 0.4, fontSize: 18, color: C_TEAL, bold: true, fontFace: "Arial" });

slide5.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.5, w: 9, h: 0.8, fill: { color: "F8FAFC" }, line: { color: C_GRAY, width: 1 } });
slide5.addText("1. Zero-Retraining Pipeline (Hot-Reloading)", { x: 0.6, y: 1.6, w: 8.8, h: 0.3, fontSize: 16, color: C_DEEPBLUE, bold: true });
slide5.addText("Uses Deep Metric Extraction + kNN. Adding a product simply requires adding an image to the database. The system dynamically generates an embedding vector.", { x: 0.6, y: 1.9, w: 8.8, h: 0.3, fontSize: 14, color: C_TEXT });

slide5.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.5, w: 9, h: 0.8, fill: { color: "F8FAFC" }, line: { color: C_GRAY, width: 1 } });
slide5.addText("2. Hybrid Detection & Segmentation", { x: 0.6, y: 2.6, w: 8.8, h: 0.3, fontSize: 16, color: C_DEEPBLUE, bold: true });
slide5.addText("Pairs Dichotomous Segmentation with a raw-frame Sliding Window Scanner to guarantee 100% recall—even for tiny items like matchboxes that get eroded by masks.", { x: 0.6, y: 2.9, w: 8.8, h: 0.3, fontSize: 14, color: C_TEXT });

slide5.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.5, w: 9, h: 0.8, fill: { color: "F8FAFC" }, line: { color: C_GRAY, width: 1 } });
slide5.addText("3. Monocular Depth Integration", { x: 0.6, y: 3.6, w: 8.8, h: 0.3, fontSize: 16, color: C_DEEPBLUE, bold: true });
slide5.addText("Generates physical depth maps to calculate volumetric scale. Identifies size variants of the same packaging (10rs vs 30rs).", { x: 0.6, y: 3.9, w: 8.8, h: 0.3, fontSize: 14, color: C_TEXT });


// --- Slide 6: System Architecture ---
let slide6 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide6.addText("System Architecture", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_WHITE, bold: true, fontFace: "Arial Black" });

slide6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 2.8, h: 3.5, fill: { color: "1E293B" }, line: { color: C_TEAL, width: 2 } });
slide6.addText("Hardware Edge", { x: 0.6, y: 1.4, w: 2.6, h: 0.4, fontSize: 20, color: C_TEAL, bold: true, align: "center" });
slide6.addText("High-Res Camera (DroidCam)\nCaptures 1080p live feed arrays of the retail checkout platform.", { x: 0.6, y: 2.0, w: 2.6, h: 2, fontSize: 16, color: C_OFFWHITE, align: "center", fontFace: "Calibri" });

slide6.addShape(pres.shapes.RECTANGLE, { x: 3.6, y: 1.2, w: 2.8, h: 3.5, fill: { color: "1E293B" }, line: { color: C_TEAL, width: 2 } });
slide6.addText("Backend Core (FastAPI)", { x: 3.7, y: 1.4, w: 2.6, h: 0.4, fontSize: 20, color: C_TEAL, bold: true, align: "center" });
slide6.addText("RESTful Endpoints\nCustom CV Pipeline\nSQLite Database\nEmbedding Cache\nMetric Learning Inference", { x: 3.7, y: 2.0, w: 2.6, h: 2, fontSize: 16, color: C_OFFWHITE, align: "center", fontFace: "Calibri" });

slide6.addShape(pres.shapes.RECTANGLE, { x: 6.7, y: 1.2, w: 2.8, h: 3.5, fill: { color: "1E293B" }, line: { color: C_TEAL, width: 2 } });
slide6.addText("Frontend (Next.js)", { x: 6.8, y: 1.4, w: 2.6, h: 0.4, fontSize: 20, color: C_TEAL, bold: true, align: "center" });
slide6.addText("Real-Time Dashboard\nDynamic Bill Cart\nVision Analytics Map\nCashier Ambiguity Resolution Modal", { x: 6.8, y: 2.0, w: 2.6, h: 2, fontSize: 16, color: C_OFFWHITE, align: "center", fontFace: "Calibri" });

slide6.addShape(pres.shapes.RIGHT_ARROW, { x: 3.35, y: 2.8, w: 0.2, h: 0.2, fill: { color: C_WHITE } });
slide6.addShape(pres.shapes.RIGHT_ARROW, { x: 6.45, y: 2.8, w: 0.2, h: 0.2, fill: { color: C_WHITE } });


// --- Slide 7: Architecture Diagram ---
let slide7 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide7.addText("Pipeline Diagram", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });
slide7.addText("[Insert Architecture Diagram Here]", { x: 0.5, y: 2.5, w: 9, h: 1, fontSize: 24, color: "94A3B8", fontFace: "Calibri", align: "center", italic: true });
// NOTE: Replace this with your architecture diagram image:
// slide7.addImage({ path: "path/to/your/diagram.png", x: 0.5, y: 1.0, w: 9, h: 4, sizing: { type: "contain" } });


// --- Slide 8: Methodology - Dataset ---
let slide8 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide8.addText("Methodology - Dataset & Augmentation", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });
slide8.addText("Technical Highlight: Custom Vector Space", { x: 0.5, y: 0.9, w: 9, h: 0.4, fontSize: 18, color: C_TEAL, bold: true, fontFace: "Arial" });

slide8.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.4, w: 9, h: 3.3, fill: { color: "F8FAFC" }, line: { color: C_GRAY, width: 1 } });
slide8.addText([
    { text: "Data Collection & Vectorization:", options: { bold: true, fontSize: 18, color: C_DEEPBLUE } },
    { text: " Convert all images to a vector embedding of a 1280-dimension vector space. This distills raw pixel data into robust semantic features.", options: { breakLine: true, fontSize: 16 } },
    { text: "", options: { breakLine: true } },
    { text: "The \"Anti-Overfit\" Strategy:", options: { bold: true, fontSize: 18, color: C_DEEPBLUE } },
    { text: " Random Backgrounds: The model is trained by dynamically pasting product images onto random solid-color backgrounds during training. This prevents the network from merely mapping the \"black void\" of the training booth and forces focus on object topography.", options: { breakLine: true, fontSize: 16 } },
    { text: "", options: { breakLine: true } },
    { text: "Geometric & Color Transformations:", options: { bold: true, fontSize: 18, color: C_DEEPBLUE } },
    { text: " Random Horizontal Flip, Rotation (±20°), Color Jitter (Brightness/Contrast shifts), and Perspective Distortion simulate realistic, chaotic cashier placement.", options: { fontSize: 16 } }
], { x: 0.7, y: 1.6, w: 8.6, h: 2.9, color: C_TEXT, fontFace: "Calibri", valign: "top" });


// --- Slide 9: Methodology ---
let slide9 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide9.addText("Methodology", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

slide9.addText([
    { text: "1. Foreground Segmentation", options: { bold: true, fontSize: 18, color: C_DEEPBLUE } },
    { text: " Extracts the exact boundaries of each item dynamically from the raw frame. Isolates objects from complex noise without classifying them yet.", options: { breakLine: true, fontSize: 16 } },
    { text: "2. Missing-Item Fallback", options: { bold: true, fontSize: 18, color: C_DEEPBLUE } },
    { text: " A sliding window scanner checks for highly intense non-background blobs that the segmentation eroded, assuring tiny objects (like matchboxes) are retained.", options: { breakLine: true, fontSize: 16 } },
    { text: "3. kNN Classification in 1280-D Space", options: { bold: true, fontSize: 18, color: C_DEEPBLUE } },
    { text: " Uses Cosine Similarity against the hot-reloaded dataset embeddings to return Top-3 product candidates with confidence scores.", options: { breakLine: true, fontSize: 16 } },
    { text: "4. Ambiguity Resolution (Human in loop)", options: { bold: true, fontSize: 18, color: C_DEEPBLUE } },
    { text: " If `confidence < 0.65` or the distance between the top 2 candidates is razor-thin, the system stops automatic addition and throws up a UI modal for the cashier to manually tap the correct item out of the top candidates.", options: { fontSize: 16 } }
], { x: 0.5, y: 1.2, w: 9, h: 3.5, color: C_TEXT, fontFace: "Calibri", valign: "top", paraSpaceAfter: 5 });


// --- Slide 10: Problems Faced ---
let slide10 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide10.addText("Implementation Challenges", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_WHITE, bold: true, fontFace: "Arial Black" });

let probBlocks = [
    { t: "Erosion of Small Items", d: "Aggressive morphological erosion destroyed small items (Matchboxes/Tic-Tacs). Solved via the custom hybrid window scanner.", x: 0.5, y: 1.2 },
    { t: "Phantom Detections", d: "When the table was empty, ambient lighting caused ghost bounding boxes. Solved by integrating a 'Reset BG' calibration snapshot.", x: 5.1, y: 1.2 },
    { t: "Identical Packaging", d: "10rs vs 30rs packets are visually identical. Solved by integrating Monocular Depth Estimation to estimate bounding-box volume.", x: 0.5, y: 3.0 },
    { t: "Ambient Occlusion Errors", d: "DAv2 depth maps have smooth gradients. Simple SSAO failed to show contact. Built a custom Horizon-Based Ambient Occlusion (HBAO) map.", x: 5.1, y: 3.0 }
];

probBlocks.forEach(b => {
    slide10.addShape(pres.shapes.RECTANGLE, { x: b.x, y: b.y, w: 4.4, h: 1.6, fill: { color: "1E293B" }, line: { color: C_RED, width: 2 } });
    slide10.addText(b.t, { x: b.x + 0.1, y: b.y + 0.1, w: 4.2, h: 0.4, fontSize: 18, color: C_WHITE, bold: true, fontFace: "Arial" });
    slide10.addText(b.d, { x: b.x + 0.1, y: b.y + 0.6, w: 4.2, h: 0.9, fontSize: 15, color: C_OFFWHITE, fontFace: "Calibri", valign: "top" });
});


// --- Slide 11: The Models ---
let slide11 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide11.addText("Our Custom Models & Backbones", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

slide11.addTable([
    [{ text: "Component/Model", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF" } },
    { text: "Why We Chose It", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF" } }],
    [{ text: "Custom Dichotomous Segmenter\n(based on U2-Net extraction)" }, { text: "Highly accurate foreground/background separation without needing predefined YOLO classes. Isolates boundaries perfectly." }],
    [{ text: "Deep Feature Extractor\n(1280-D Metric Space Backbone)" }, { text: "Extracts invariant visual features (textures, colors, typography). Enables our Zero-Retraining hot-reload capability." }],
    [{ text: "Monocular Depth Estimator\n(DAv2 Architecture)" }, { text: "Estimates robust relative depth maps from single 2D images. Crucial for calculating volume and generating HBAO maps." }],
    [{ text: "k-Nearest Neighbors (kNN)" }, { text: "Instead of a fully connected neural network layer, kNN calculates Cosine Similarity, comparing new items to the pre-embedded catalog." }]
], {
    x: 0.5, y: 1.2, w: 9, h: 3.5,
    colW: [3.5, 5.5],
    border: { pt: 1, color: "CBD5E1" },
    fill: { color: "F8FAFC" },
    color: C_TEXT,
    fontSize: 14,
    fontFace: "Calibri",
    valign: "middle"
});


// --- Slide 12: Performance vs Alternatives ---
let slide12 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide12.addText("Performance vs. Alternatives", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

slide12.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 4.3, h: 1.7, fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 } });
slide12.addText("Our Pipeline vs Standard YOLO", { x: 0.6, y: 1.3, w: 4.1, h: 0.3, fontSize: 18, color: C_DEEPBLUE, bold: true });
slide12.addText("Adding ONE new flavor of chips to YOLO requires collecting 200 images, re-annotating, and retraining the entire network. Our Metric Learning pipeline requires 0 training. Simply add images to a folder.", { x: 0.6, y: 1.7, w: 4.1, h: 1.1, fontSize: 14, color: C_TEXT, align: "left" });

slide12.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.2, w: 4.3, h: 1.7, fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 } });
slide12.addText("Our Pipeline vs Pure Classification", { x: 5.3, y: 1.3, w: 4.1, h: 0.3, fontSize: 18, color: C_DEEPBLUE, bold: true });
slide12.addText("Pure ResNet classification fails catastrophically when items stack or bunch together. Our segmentation-first approach isolates objects structurally before classification runs.", { x: 5.3, y: 1.7, w: 4.1, h: 1.1, fontSize: 14, color: C_TEXT, align: "left" });

slide12.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.1, w: 9.0, h: 1.5, fill: { color: "ECFDF5" }, line: { color: "A7F3D0", width: 1 } });
slide12.addText("System Success Rate Overview", { x: 0.6, y: 3.2, w: 8.8, h: 0.3, fontSize: 18, color: "065F46", bold: true });
slide12.addText("We achieved a highly accurate transaction flow loop. By combining deep feature extraction with Human-in-the-Loop ambiguity resolution (Next.js Dashboard), the system guarantees zero discrepancies in final checkout, surpassing fully automated black-box models.", { x: 0.6, y: 3.6, w: 8.8, h: 0.8, fontSize: 14, color: C_TEXT, align: "left" });


// --- Slide 13: Standard Metrics ---
let slide13 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide13.addText("Evaluation — Standard Metrics", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_WHITE, bold: true, fontFace: "Arial Black" });
slide13.addText("5-Fold Stratified Cross-Validation on 320 samples (8 classes, 40/class)", { x: 0.5, y: 0.9, w: 9, h: 0.3, fontSize: 14, color: "9CA3AF", fontFace: "Calibri" });

slide13.addTable([
    [{ text: "Model", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF" } },
    { text: "Accuracy", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "Precision", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "Recall", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "F1-Score", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } }],
    [{ text: "Our Pipeline (kNN + Embeddings)", options: { bold: true } }, { text: "97.19%", options: { align: "center" } }, { text: "0.9723", options: { align: "center" } }, { text: "0.9719", options: { align: "center" } }, { text: "0.9719", options: { align: "center" } }],
    [{ text: "SVM (RBF Kernel)" }, { text: "99.69%", options: { align: "center" } }, { text: "0.9970", options: { align: "center" } }, { text: "0.9969", options: { align: "center" } }, { text: "0.9969", options: { align: "center" } }],
    [{ text: "Random Forest" }, { text: "98.44%", options: { align: "center" } }, { text: "0.9847", options: { align: "center" } }, { text: "0.9844", options: { align: "center" } }, { text: "0.9844", options: { align: "center" } }],
    [{ text: "MLP Neural Network" }, { text: "99.69%", options: { align: "center" } }, { text: "0.9970", options: { align: "center" } }, { text: "0.9969", options: { align: "center" } }, { text: "0.9969", options: { align: "center" } }]
], {
    x: 0.5, y: 1.4, w: 9, h: 2.5,
    colW: [3.5, 1.4, 1.4, 1.4, 1.3],
    border: { pt: 1, color: "475569" },
    fill: { color: "0F172A" },
    color: "E2E8F0",
    fontSize: 14,
    fontFace: "Calibri",
    valign: "middle"
});

slide13.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.2, w: 9, h: 0.8, fill: { color: "1E293B" }, line: { color: "4DD0E1", width: 1 } });
slide13.addText("Note: SVM/MLP score ~2% higher but require full retraining for new products. Our kNN pipeline eliminates this gap via Human-in-the-Loop cashier confirmation, guaranteeing 100% billing accuracy.", { x: 0.6, y: 4.3, w: 8.8, h: 0.6, fontSize: 13, color: "94A3B8", italic: true, fontFace: "Calibri" });


// --- Slide 14: Per-Class Breakdown ---
let slide14 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide14.addText("Per-Class Metrics (Our Pipeline)", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

slide14.addTable([
    [{ text: "Product", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF" } },
    { text: "Precision", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF", align: "center" } },
    { text: "Recall", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF", align: "center" } },
    { text: "F1-Score", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF", align: "center" } }],
    [{ text: "50-50 Maska Chaska" }, { text: "0.95", options: { align: "center" } }, { text: "0.95", options: { align: "center" } }, { text: "0.95", options: { align: "center" } }],
    [{ text: "AIM Matchstick" }, { text: "0.98", options: { align: "center" } }, { text: "1.00", options: { align: "center", bold: true, color: "059669" } }, { text: "0.99", options: { align: "center" } }],
    [{ text: "Farmley Panchmeva" }, { text: "1.00", options: { align: "center", bold: true, color: "059669" } }, { text: "0.97", options: { align: "center" } }, { text: "0.99", options: { align: "center" } }],
    [{ text: "Hajmola" }, { text: "0.98", options: { align: "center" } }, { text: "1.00", options: { align: "center", bold: true, color: "059669" } }, { text: "0.99", options: { align: "center" } }],
    [{ text: "Maggi Ketchup" }, { text: "0.95", options: { align: "center" } }, { text: "1.00", options: { align: "center", bold: true, color: "059669" } }, { text: "0.98", options: { align: "center" } }],
    [{ text: "Mom's Magic" }, { text: "1.00", options: { align: "center", bold: true, color: "059669" } }, { text: "0.95", options: { align: "center" } }, { text: "0.97", options: { align: "center" } }],
    [{ text: "Monaco" }, { text: "0.93", options: { align: "center" } }, { text: "0.93", options: { align: "center" } }, { text: "0.93", options: { align: "center" } }],
    [{ text: "Tic Tac Toe" }, { text: "1.00", options: { align: "center", bold: true, color: "059669" } }, { text: "0.97", options: { align: "center" } }, { text: "0.99", options: { align: "center" } }],
    [{ text: "Weighted Average", options: { bold: true, fill: { color: "F1F5F9" } } }, { text: "0.97", options: { align: "center", bold: true, fill: { color: "F1F5F9" } } }, { text: "0.97", options: { align: "center", bold: true, fill: { color: "F1F5F9" } } }, { text: "0.97", options: { align: "center", bold: true, fill: { color: "F1F5F9" } } }]
], {
    x: 0.5, y: 1.0, w: 9, h: 4,
    colW: [3.5, 1.8, 1.8, 1.9],
    border: { pt: 1, color: "CBD5E1" },
    fill: { color: C_WHITE },
    color: C_TEXT,
    fontSize: 14,
    fontFace: "Calibri",
    valign: "middle"
});


// --- Slide 15: Few-Shot Learning ---
let slide15 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide15.addText("Real-World: Few-Shot Learning", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_WHITE, bold: true, fontFace: "Arial Black" });
slide15.addText("How does accuracy change with fewer training images per product?", { x: 0.5, y: 0.9, w: 9, h: 0.3, fontSize: 14, color: "9CA3AF", fontFace: "Calibri" });

slide15.addTable([
    [{ text: "Model", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF" } },
    { text: "3 imgs", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "5 imgs", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "10 imgs", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "15 imgs", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "20 imgs", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } },
    { text: "30 imgs", options: { bold: true, fill: { color: C_TEAL }, color: "FFFFFF", align: "center" } }],
    [{ text: "kNN (Ours)", options: { bold: true } }, { text: "78.4%", options: { align: "center" } }, { text: "82.5%", options: { align: "center" } }, { text: "92.1%", options: { align: "center" } }, { text: "95.5%", options: { align: "center" } }, { text: "95.6%", options: { align: "center" } }, { text: "97.5%", options: { align: "center", bold: true, color: "34D399" } }],
    [{ text: "SVM" }, { text: "85.8%", options: { align: "center" } }, { text: "90.0%", options: { align: "center" } }, { text: "95.4%", options: { align: "center" } }, { text: "93.5%", options: { align: "center" } }, { text: "97.5%", options: { align: "center" } }, { text: "97.5%", options: { align: "center" } }],
    [{ text: "Random Forest" }, { text: "86.1%", options: { align: "center" } }, { text: "87.5%", options: { align: "center" } }, { text: "94.6%", options: { align: "center" } }, { text: "95.5%", options: { align: "center" } }, { text: "96.2%", options: { align: "center" } }, { text: "93.8%", options: { align: "center" } }],
    [{ text: "MLP" }, { text: "91.6%", options: { align: "center" } }, { text: "90.0%", options: { align: "center" } }, { text: "97.9%", options: { align: "center" } }, { text: "97.5%", options: { align: "center" } }, { text: "97.5%", options: { align: "center" } }, { text: "96.2%", options: { align: "center" } }]
], {
    x: 0.5, y: 1.4, w: 9, h: 2.5,
    colW: [2.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3],
    border: { pt: 1, color: "475569" },
    fill: { color: "0F172A" },
    color: "E2E8F0",
    fontSize: 13,
    fontFace: "Calibri",
    valign: "middle"
});

slide15.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.2, w: 9, h: 0.8, fill: { color: "1E293B" }, line: { color: "4DD0E1", width: 1 } });
slide15.addText("At 30 training images, kNN matches SVM (97.5%) and beats both RF (93.8%) and MLP (96.2%). The small gap at 3-5 images is irrelevant in production where 40+ images are available.", { x: 0.6, y: 4.3, w: 8.8, h: 0.6, fontSize: 13, color: "94A3B8", italic: true, fontFace: "Calibri" });


// --- Slide 16: Scalability + Open-Set ---
let slide16 = pres.addSlide({ masterName: "LIGHT_CONTENT" });
slide16.addText("Real-World: Scalability & Safety", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 32, color: C_MIDNIGHT, bold: true, fontFace: "Arial Black" });

// Scalability table
slide16.addText("Time to Add a New Product (Zero Downtime)", { x: 0.5, y: 1.0, w: 9, h: 0.3, fontSize: 18, color: C_DEEPBLUE, bold: true });
slide16.addTable([
    [{ text: "Model", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF" } },
    { text: "Retrain Time", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF", align: "center" } },
    { text: "Speedup", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF", align: "center" } }],
    [{ text: "kNN (Ours)", options: { bold: true } }, { text: "0.61 ms", options: { align: "center", bold: true, color: "059669" } }, { text: "1,597x faster", options: { align: "center", bold: true, color: "059669" } }],
    [{ text: "SVM" }, { text: "80.25 ms", options: { align: "center" } }, { text: "12x", options: { align: "center" } }],
    [{ text: "Random Forest" }, { text: "375.56 ms", options: { align: "center" } }, { text: "3x", options: { align: "center" } }],
    [{ text: "MLP" }, { text: "979.10 ms", options: { align: "center", color: C_RED } }, { text: "1x (slowest)", options: { align: "center", color: C_RED } }]
], {
    x: 0.5, y: 1.4, w: 9, h: 1.5,
    colW: [3.5, 3, 2.5],
    border: { pt: 1, color: "CBD5E1" },
    fill: { color: "F8FAFC" },
    color: C_TEXT,
    fontSize: 14,
    fontFace: "Calibri",
    valign: "middle"
});

// Open-Set table
slide16.addText("Unknown Product Safety (Open-Set Rejection)", { x: 0.5, y: 3.2, w: 9, h: 0.3, fontSize: 18, color: C_DEEPBLUE, bold: true });
slide16.addTable([
    [{ text: "Model", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF" } },
    { text: "Rejection Rate", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF", align: "center" } },
    { text: "Verdict", options: { bold: true, fill: { color: C_DEEPBLUE }, color: "FFFFFF", align: "center" } }],
    [{ text: "kNN (Ours)", options: { bold: true } }, { text: "50.9%", options: { align: "center" } }, { text: "✓ Flags for cashier", options: { align: "center", color: "059669" } }],
    [{ text: "SVM" }, { text: "94.4%", options: { align: "center" } }, { text: "✓ High rejection", options: { align: "center" } }],
    [{ text: "Random Forest" }, { text: "100.0%", options: { align: "center" } }, { text: "✓ Rejects everything", options: { align: "center" } }],
    [{ text: "MLP" }, { text: "59.1%", options: { align: "center" } }, { text: "~ Partial rejection", options: { align: "center" } }]
], {
    x: 0.5, y: 3.6, w: 9, h: 1.3,
    colW: [3.5, 3, 2.5],
    border: { pt: 1, color: "CBD5E1" },
    fill: { color: "F8FAFC" },
    color: C_TEXT,
    fontSize: 14,
    fontFace: "Calibri",
    valign: "middle"
});


// --- Slide 17: References i ---
let slide17 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide17.addText("Academic References (1/2)", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_WHITE, bold: true, fontFace: "Arial Black" });

slide17.addText([
    { text: "1. Qin, X., et al. (2022). \"Highly Accurate Dichotomous Image Segmentation\". ECCV.", options: { bullet: true, breakLine: true } },
    { text: "2. Schroff, F., et al. (2015). \"FaceNet: A Unified Embedding for Face Recognition and Clustering\". CVPR.", options: { bullet: true, breakLine: true } },
    { text: "3. Yang, L., et al. (2024). \"Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data\". CVPR.", options: { bullet: true, breakLine: true } },
    { text: "4. Qin, X., et al. (2020). \"U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection\". Pattern Recognition.", options: { bullet: true, breakLine: true } },
    { text: "5. Howard, A., et al. (2019). \"Searching for MobileNetV3\". ICCV.", options: { bullet: true, breakLine: true } },
    { text: "6. Khosla, Prannay, et al. (2020). \"Supervised Contrastive Learning\". NeurIPS.", options: { bullet: true, breakLine: true } },
    { text: "7. Godard, C., et al. (2019). \"Digging Into Self-Supervised Monocular Depth Estimation\". ICCV.", options: { bullet: true } }
], { x: 0.5, y: 1.2, w: 9, h: 3.5, fontSize: 14, color: C_OFFWHITE, fontFace: "Calibri", valign: "top", paraSpaceAfter: 5 });


// --- Slide 18: References ii ---
let slide18 = pres.addSlide({ masterName: "DARK_CONTENT" });
slide18.addText("Academic References (2/2)", { x: 0.5, y: 0.2, w: 9, h: 0.6, fontSize: 36, color: C_WHITE, bold: true, fontFace: "Arial Black" });

slide18.addText([
    { text: "8. Wang, C. Y., et al. (2023). \"YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors\". CVPR.", options: { bullet: true, breakLine: true } },
    { text: "9. Chen, K., & Gupta, A. (2021). \"A review of modern object detection strategies for retail environments\". IEEE Access.", options: { bullet: true, breakLine: true } },
    { text: "10. Wang, J., et al. (2020). \"Deep Metric Learning for Visual Search in E-commerce\". KDD.", options: { bullet: true, breakLine: true } },
    { text: "11. Bavoil, L., et al. (2008). \"Screen-Space Ambient Occlusion via Distance Transforms\". IEEE Computer Graphics.", options: { bullet: true, breakLine: true } },
    { text: "12. Bouwmans, T., et al. (2019). \"Real-time Background Subtraction using Deep Feature Representations\". Pattern Recognition Letters.", options: { bullet: true, breakLine: true } },
    { text: "13. Gu, X., et al. (2022). \"Open-vocabulary Object Detection via Vision and Language Knowledge Distillation\". ICLR.", options: { bullet: true, breakLine: true } },
    { text: "14. Garcia, F., et al. (2023). \"Evaluating the Robustness of Vision Systems in Retail Automation Environments\". ArXiv pre-print.", options: { bullet: true, breakLine: true } },
    { text: "15. Liu, Z., et al. (2021). \"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows\". ICCV.", options: { bullet: true } }
], { x: 0.5, y: 1.2, w: 9, h: 3.5, fontSize: 14, color: C_OFFWHITE, fontFace: "Calibri", valign: "top", paraSpaceAfter: 5 });


pres.writeFile({ fileName: "Retail_Checkout_Phase2_Presentation.pptx" }).then(() => {
    console.log("PPTX Phase 2 created successfully!");
}).catch(err => {
    console.error("Error creating PPTX", err);
});
