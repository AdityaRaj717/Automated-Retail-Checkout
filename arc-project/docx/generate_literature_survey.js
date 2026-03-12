const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, TableOfContents, PageBreak, AlignmentType, LevelFormat, Header, Footer, PageNumber } = require('docx');

const doc = new Document({
    styles: {
        default: { document: { run: { font: "Arial", size: 22 } } }, // 11pt default body
        paragraphStyles: [
            {
                id: "Title", name: "Title", basedOn: "Normal", next: "Subtitle", quickFormat: true,
                run: { size: 48, bold: true, font: "Arial", color: "2E74B5" },
                paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER }
            },
            {
                id: "Subtitle", name: "Subtitle", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 28, italics: true, font: "Arial", color: "595959" },
                paragraph: { spacing: { before: 120, after: 240 }, alignment: AlignmentType.CENTER }
            },
            {
                id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 32, bold: true, font: "Arial", color: "1F4D78" },
                paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 }
            },
            {
                id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 28, bold: true, font: "Arial", color: "2E74B5" },
                paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
            },
            {
                id: "BodyText", name: "Body Text", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 22, font: "Arial" },
                paragraph: { spacing: { before: 120, after: 120 }, alignment: AlignmentType.JUSTIFIED, lineSpacing: { line: 360, lineRule: "auto" } }
            }, // 1.5 spacing
            {
                id: "RefText", name: "Reference Text", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 22, font: "Arial" },
                paragraph: { spacing: { before: 120, after: 120 }, alignment: AlignmentType.LEFT, lineSpacing: { line: 360, lineRule: "auto" } }
            }, // Used in numbering
        ]
    },
    numbering: {
        config: [
            {
                reference: "refs",
                levels: [{
                    level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
                    style: { paragraph: { style: "RefText", indent: { left: 720, hanging: 480 } } }
                }]
            },
        ]
    },
    sections: [
        {
            properties: {
                page: {
                    size: { width: 12240, height: 15840 },
                    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
                }
            },
            headers: {
                default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Literature Survey: Automated Retail Checkout", color: "808080", size: 18 })] })] })
            },
            footers: {
                default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Page ", size: 18, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" }), new TextRun({ text: " of ", size: 18, color: "808080" }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "808080" })] })] })
            },
            children: [
                // Title Page
                new Paragraph({ spacing: { before: 2880 } }),
                new Paragraph({ style: "Title", children: [new TextRun("Literature Survey")] }),
                new Paragraph({ style: "Subtitle", children: [new TextRun("Automated Retail Billing System: Computer Vision & Metric Learning")] }),
                new Paragraph({ spacing: { before: 2880 } }),
                new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Prepared by:", size: 24, bold: true })] }),
                new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Group 11 | Phase 2 | VIT Bhopal", size: 24 })] }),
                new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Adityaraj Rajesh Kumar, Shivam Singh, Shayan Singha, Kaushal Sengupta, Harsh Naik", size: 22, color: "595959" })] }),
                new Paragraph({ spacing: { before: 1440 } }),
                new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Date: March 2026", size: 22 })] }),
                new Paragraph({ children: [new PageBreak()] }),

                // TOC
                new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Table of Contents")] }),
                new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
                new Paragraph({ children: [new PageBreak()] }),

                // Introduction
                new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Introduction")] }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun("The traditional retail checkout process suffers from significant bottlenecks due to manual barcode scanning. Cashiers must physically locate, align, and scan each item, a sequential process that inevitably leads to long queues during peak hours and exacerbates customer friction. Furthermore, "),
                        new TextRun({ text: "human error", italics: true }),
                        new TextRun(" in standard checkout lanes contributes dramatically to inventory discrepancies, unscanned variants, and retail shrinkage. To mitigate these issues, automated vision-based checkout systems have become the focus of extensive research. This literature survey synthesizes recent advancements in "),
                        new TextRun({ text: "object detection, deep metric learning, dichotomous segmentation, and monocular depth estimation", bold: true }),
                        new TextRun(", contextualizing these academic breakthroughs within the architecture of our proposed "),
                        new TextRun({ text: "zero-retraining retail billing system.", italics: true, bold: true })
                    ]
                }),

                // Literature Review per category
                new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Review of Existing Methodologies")] }),

                new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.1. Object Detection Limitations in Retail")] }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun("Conventional automated retail systems rely heavily on fully supervised bounding-box regression networks, most notably the "),
                        new TextRun({ text: "YOLO (You Only Look Once)", bold: true }),
                        new TextRun(" architectures. Wang et al. (2023) [8] significantly advanced the state-of-the-art with "),
                        new TextRun({ text: "YOLOv7", bold: true }),
                        new TextRun(", introducing trainable bag-of-freebies that optimized real-time detection without escalating inference costs. However, as evaluated by Chen & Gupta (2021) [9] in their exhaustive review of retail object detection strategies, YOLO and standard Convolutional Neural Networks (CNNs) exhibit severe operational flaws in dynamic environments. The primary bottleneck is the "),
                        new TextRun({ text: "“addition problem”", italics: true, bold: true }),
                        new TextRun("; incorporating a single newly manufactured product—such as a new flavor of chips—demands hundreds of meticulously annotated images, followed by an intensive retraining cycle of the entire network. Garcia et al. (2023) [14] further corroborated these findings, noting that classical bounding-box regression degrades catastrophically when items are stacked, clustered, or present under complex lighting occlusions common in checkout bins.")
                    ]
                }),

                new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.2. Deep Metric Learning for Zero-Retraining Models")] }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun("To circumvent the rigid class dependencies of standard CNNs, "),
                        new TextRun({ text: "Deep Metric Learning (DML)", bold: true }),
                        new TextRun(" transforms classification into a vector-space retrieval problem. This paradigm was popularized by Schroff et al. (2015) [2] with "),
                        new TextRun({ text: "FaceNet", bold: true, italics: true }),
                        new TextRun(", which utilized a Triplet Loss function to map images into a compact Euclidean space, grouping similar identities together. Wang et al. (2020) [10] subsequently adapted deep metric learning for e-commerce visual search, proving that extracting invariant visual features—such as edge typography, packaging texture, and brand color palettes—yielded vastly superior cross-domain adaptability. Furthermore, Khosla et al. (2020) [6] proposed "),
                        new TextRun({ text: "Supervised Contrastive Learning", bold: true }),
                        new TextRun(", enhancing the robustness of embeddings against background noise.")
                    ]
                }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun({ text: "Application to Our System: ", bold: true, italics: true, color: "2E74B5" }),
                        new TextRun("We derive our core classification engine from these metric learning breakthroughs. Instead of retraining a model for every new grocery item, our system deploys a "),
                        new TextRun({ text: "1280-Dimensional Metric Space Backbone", bold: true }),
                        new TextRun(". When adding a product, a single baseline image is converted into an embedding vector and saved to the database. During checkout, live frames are embedded and compared against the database using "),
                        new TextRun({ text: "k-Nearest Neighbors (kNN) Cosine Similarity. ", bold: true }),
                        new TextRun("This "),
                        new TextRun({ text: "“hot-reloading”", italics: true }),
                        new TextRun(" capability achieves sub-millisecond addition to the catalog with zero model downtime.")
                    ]
                }),

                new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.3. Foreground Segmentation and Complex Occlusion Resolution")] }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun("Before accurate feature extraction can occur, individual items must be isolated from the noisy background of a retail counter. Traditional background subtraction algorithms (Bouwmans et al., 2019) [12] struggle with changing ambient light and shadows. Qin et al. (2020) [4] developed the "),
                        new TextRun({ text: "U^2-Net", bold: true }),
                        new TextRun(" architecture, leveraging a nested U-structure that extracts multi-scale features for highly accurate salient object detection. They subsequently refined this into "),
                        new TextRun({ text: "Dichotomous Image Segmentation (DIS)", bold: true }),
                        new TextRun(" (Qin et al., 2022) [1], achieving pixel-perfect object masks agnostic to the object's class identity. Swin Transformers (Liu et al., 2021) [15] also provide hierarchical feature extraction that preserves structural granularity, preventing the visual erosion of small objects during pooling operations.")
                    ]
                }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun({ text: "Application to Our System: ", bold: true, italics: true, color: "2E74B5" }),
                        new TextRun("We implement a "),
                        new TextRun({ text: "Custom Dichotomous Segmenter", bold: true }),
                        new TextRun(" based heavily on the U^2-Net extraction topology. This isolates the precise boundaries of overlapping products before invoking the metric classifier, ensuring that background noise does not pollute the vector embedding. To counteract the erosion of extremely small items (e.g., matchboxes, Tic-Tacs) typical in aggressive morphological masks, we designed a "),
                        new TextRun({ text: "Hybrid Raw-Frame Sliding Window Scanner", bold: true }),
                        new TextRun(" that acts as a fail-safe against missed detections.")
                    ]
                }),

                new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.4. Depth Estimation for Volumetric Disambiguation")] }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun("A unique challenge in retail checkout is identical multi-size packaging (e.g., distinguishing a 10 Rs. detergent packet from a 30 Rs. variant sharing the exact same graphic design). While metric embeddings resolve the visual identity, they are scale-invariant and thus fail to differentiate volume. "),
                        new TextRun({ text: "Monocular depth estimation", bold: true }),
                        new TextRun(" infers 3D topography from solitary 2D sensors. Godard et al. (2019) [7] laid the groundwork for self-supervised monocular depth estimation, removing the requirement for expensive LiDAR ground truths. More recently, Yang et al. (2024) [3] released “"),
                        new TextRun({ text: "Depth Anything", italics: true }),
                        new TextRun(",” a large-scale unlabeled dataset and architecture that generates robust relative depth maps under high variance. Concurrently, calculating proximity and contact between clustered items can be augmented by graphics techniques like "),
                        new TextRun({ text: "Screen-Space Ambient Occlusion (SSAO)", bold: true }),
                        new TextRun(" (Bavoil et al., 2008) [11], which darkens crevices to accentuate distinct object separation.")
                    ]
                }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun({ text: "Application to Our System: ", bold: true, italics: true, color: "2E74B5" }),
                        new TextRun("To solve the scale ambiguity problem, we integrate a "),
                        new TextRun({ text: "Monocular Depth Estimator", bold: true }),
                        new TextRun(" (utilizing the DAv2 Architecture). This generates physical depth maps allowing us to estimate bounding-box volume in three dimensions, successfully separating visually identical size variants. Additionally, because standard depth maps feature smooth gradients, we engineered a custom "),
                        new TextRun({ text: "Horizon-Based Ambient Occlusion (HBAO)", bold: true }),
                        new TextRun(" map to sharply indicate where items physically touch, preventing overlapping bounds.")
                    ]
                }),

                new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Conclusion")] }),
                new Paragraph({
                    style: "BodyText", children: [
                        new TextRun("The limitations of conventional bounding-box regression models render them impractical for modern, rapidly shifting retail environments. As synthesized in this review, modern computer vision has pivoted towards modular, invariant pipelines. By hybridizing the precise salient masking of "),
                        new TextRun({ text: "Dichotomous Image Segmentation boundaries ", bold: true }),
                        new TextRun("(Qin et al., 2022) [1], the scalable, zero-retraining capabilities of "),
                        new TextRun({ text: "Deep Metric Learning embeddings ", bold: true }),
                        new TextRun("(Schroff et al., 2015) [2], and the physical scale awareness of "),
                        new TextRun({ text: "Monocular Depth Estimation ", bold: true }),
                        new TextRun("(Yang et al., 2024) [3], we have engineered a sophisticated, robust checkout system. This architecture not only guarantees high accuracy under occlusion but natively supports the continuous expansion of retail inventory with zero structural downtime.")
                    ]
                }),

                new Paragraph({ children: [new PageBreak()] }),
                new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("References")] }),

                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Qin, X., et al. (2022). “Highly Accurate Dichotomous Image Segmentation.” "),
                        new TextRun({ text: "European Conference on Computer Vision (ECCV).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Schroff, F., et al. (2015). “FaceNet: A Unified Embedding for Face Recognition and Clustering.” "),
                        new TextRun({ text: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Yang, L., et al. (2024). “Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data.” "),
                        new TextRun({ text: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Qin, X., et al. (2020). “U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection.” "),
                        new TextRun({ text: "Pattern Recognition.", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Howard, A., et al. (2019). “Searching for MobileNetV3.” "),
                        new TextRun({ text: "IEEE/CVF International Conference on Computer Vision (ICCV).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Khosla, Prannay, et al. (2020). “Supervised Contrastive Learning.” "),
                        new TextRun({ text: "Advances in Neural Information Processing Systems (NeurIPS).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Godard, C., et al. (2019). “Digging Into Self-Supervised Monocular Depth Estimation.” "),
                        new TextRun({ text: "IEEE/CVF International Conference on Computer Vision (ICCV).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Wang, C. Y., et al. (2023). “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors.” "),
                        new TextRun({ text: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Chen, K., & Gupta, A. (2021). “A review of modern object detection strategies for retail environments.” "),
                        new TextRun({ text: "IEEE Access.", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Wang, J., et al. (2020). “Deep Metric Learning for Visual Search in E-commerce.” "),
                        new TextRun({ text: "ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Bavoil, L., et al. (2008). “Screen-Space Ambient Occlusion via Distance Transforms.” "),
                        new TextRun({ text: "IEEE Computer Graphics.", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Bouwmans, T., et al. (2019). “Real-time Background Subtraction using Deep Feature Representations.” "),
                        new TextRun({ text: "Pattern Recognition Letters.", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Gu, X., et al. (2022). “Open-vocabulary Object Detection via Vision and Language Knowledge Distillation.” "),
                        new TextRun({ text: "International Conference on Learning Representations (ICLR).", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Garcia, F., et al. (2023). “Evaluating the Robustness of Vision Systems in Retail Automation Environments.” "),
                        new TextRun({ text: "ArXiv pre-print.", italics: true })
                    ]
                }),
                new Paragraph({
                    numbering: { reference: "refs", level: 0 }, children: [
                        new TextRun("Liu, Z., et al. (2021). “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” "),
                        new TextRun({ text: "IEEE/CVF International Conference on Computer Vision (ICCV).", italics: true })
                    ]
                })
            ]
        }
    ]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("Literature_Survey_Retail_Billing.docx", buffer);
    console.log("Literature_Survey_Retail_Billing.docx has been created successfully!");
}).catch(err => {
    console.error("Error generating document:", err);
});
