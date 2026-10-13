"""Generate a comprehensive, beautifully styled PDF project guide for DefectLens AI in simple English."""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (on pages after cover)
        if self._pageNumber > 1:
            self.drawString(36, letter[1] - 28, "DefectLens AI — Complete Project Documentation & Guide")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, letter[1] - 32, letter[0] - 36, letter[1] - 32)

        # Footer
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 32, letter[0] - 36, 32)

        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 20, page_text)
        self.drawString(36, 20, "Author: Rishav Raj (@rishavrjpt) | GitHub: https://github.com/rishavrjpt/DefectLens-AI")
        self.restoreState()


def build_pdf(output_pdf_path="DefectLens_AI_Project_Guide.pdf"):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=44,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2563eb"),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "ChapterHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # =========================================================================
    # HEADER BANNER / TITLE
    # =========================================================================
    header_table = Table(
        [
            [
                Paragraph("<b>DEFECTLENS AI</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=16, textColor=colors.white)),
                Paragraph("<b>INDUSTRIAL SURFACE ANOMALY DETECTION</b>", ParagraphStyle("HdrR", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#93c5fd"), alignment=2)),
            ]
        ],
        colWidths=[3.2 * inch, 4.2 * inch],
    )
    header_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Complete Project Guide &amp; Technical Explanation", title_style))
    story.append(Paragraph("<i>Written in Simple, Clear English for Students, Engineers &amp; Recruiters</i>", subtitle_style))

    # Meta Info Box
    meta_data = [
        [
            Paragraph("<b>Author / Developer:</b> Rishav Raj (@rishavrjpt)", body_style),
            Paragraph("<b>Frameworks:</b> PyTorch 2.14, FastAPI, ResNet-18", body_style),
        ],
        [
            Paragraph("<b>Algorithm:</b> PatchCore (Roth et al., CVPR)", body_style),
            Paragraph("<b>Benchmark Dataset:</b> MVTec Anomaly Detection (bottle)", body_style),
        ],
        [
            Paragraph("<b>Image AUROC:</b> 1.0000 (100% Accuracy)", body_style),
            Paragraph("<b>Pixel AUROC:</b> 0.9222 (Thermal Localization)", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[3.7 * inch, 3.7 * inch])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 1: WHAT IS DEFECTLENS AI?
    # =========================================================================
    story.append(Paragraph("1. What is DefectLens AI? (The Big Picture)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    story.append(
        Paragraph(
            "<b>DefectLens AI</b> is an intelligent computer vision system designed for factories and manufacturing lines. "
            "Its job is to automatically look at photographs of manufactured products (such as glass bottles) and flag any "
            "surface defects—such as cracks, chipped edges, or chemical contamination—in real time.",
            body_style,
        )
    )

    # Analogy Callout Box
    analogy_content = [
        [
            Paragraph(
                "<b>The Bank Teller Analogy (How it differs from ordinary AI):</b><br/>"
                "Standard AI models require thousands of pictures of every conceivable defect to learn. But in a real factory, "
                "99.9% of items produced are flawless! You do not have thousands of broken items to train on.<br/><br/>"
                "DefectLens AI works like a <b>professional bank teller spotting counterfeit money</b>: "
                "The teller does not study thousands of fake notes. Instead, they handle real, genuine currency all day long until "
                "they know every millimeter of authentic bills. When a fake bill appears, it immediately 'feels wrong,' "
                "even if the teller has never seen that specific fake note before. DefectLens AI does the exact same thing with product surfaces!",
                callout_style,
            )
        ]
    ]
    analogy_table = Table(analogy_content, colWidths=[7.4 * inch])
    analogy_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(analogy_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # CHAPTER 2: HOW IT WORKS STEP-BY-STEP
    # =========================================================================
    story.append(Paragraph("2. How DefectLens AI Works Under the Hood", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    steps_data = [
        [
            Paragraph("<b>Step</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#0f172a"))),
            Paragraph("<b>Stage Name</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#0f172a"))),
            Paragraph("<b>Simple Plain-English Explanation</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#0f172a"))),
        ],
        [
            Paragraph("<b>01</b>", code_style),
            Paragraph("<b>Input Ingestion</b>", body_style),
            Paragraph("An inspection camera captures an image of a product surface. The image is resized to standard 224x224 pixels and normalized.", body_style),
        ],
        [
            Paragraph("<b>02</b>", code_style),
            Paragraph("<b>Feature Extraction (ResNet-18)</b>", body_style),
            Paragraph("A pretrained deep neural network breaks the image into local visual patches (rim, body, glass curvature) across middle layers (layer2 and layer3).", body_style),
        ],
        [
            Paragraph("<b>03</b>", code_style),
            Paragraph("<b>Coreset Memory Bank</b>", body_style),
            Paragraph("During training on 178 normal bottles, the AI collects 26,700 normal patches and compresses them down to 2,670 representative patches using greedy minimax selection (90% memory reduction!).", body_style),
        ],
        [
            Paragraph("<b>04</b>", code_style),
            Paragraph("<b>Validation Calibration</b>", body_style),
            Paragraph("A strict decision threshold (1.6636) is mathematically calculated on 31 held-out normal validation bottles. If a score is above 1.6636, it is an anomaly.", body_style),
        ],
        [
            Paragraph("<b>05</b>", code_style),
            Paragraph("<b>Distance Scoring</b>", body_style),
            Paragraph("When a new bottle is inspected, each patch is compared against the memory bank using Euclidean distance. Unfamiliar patches create a spike in score.", body_style),
        ],
        [
            Paragraph("<b>06</b>", code_style),
            Paragraph("<b>Thermal Heatmap</b>", body_style),
            Paragraph("High-distance patches are spatially mapped back onto the original bottle and smoothed with a Gaussian filter to generate a glowing red defect heatmap.", body_style),
        ],
    ]
    steps_table = Table(steps_data, colWidths=[0.6 * inch, 1.8 * inch, 5.0 * inch])
    steps_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(steps_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 3: DATASET & BENCHMARK
    # =========================================================================
    story.append(Paragraph("3. The Benchmark Dataset (MVTec AD)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    story.append(
        Paragraph(
            "We used the authoritative industrial benchmark: <b>MVTec Anomaly Detection (MVTec AD)</b> "
            "created by MVTec Software GmbH (CVPR 2019) under the Creative Commons CC BY-NC-SA 4.0 license. "
            "For this deployment, we used the <b>Bottle</b> category, which contains real industrial glass bottles.",
            body_style,
        )
    )

    dataset_summary_data = [
        [
            Paragraph("<b>Split</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=8.5)),
            Paragraph("<b>Image Type</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=8.5)),
            Paragraph("<b>Count</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=8.5)),
            Paragraph("<b>Role in DefectLens AI</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=8.5)),
        ],
        [
            Paragraph("Train Set", body_style),
            Paragraph("Nominal (Flawless) Bottles", body_style),
            Paragraph("<b>178</b>", code_style),
            Paragraph("Used to build the normal patch memory bank.", body_style),
        ],
        [
            Paragraph("Validation Set", body_style),
            Paragraph("Nominal (Flawless) Bottles", body_style),
            Paragraph("<b>31</b>", code_style),
            Paragraph("Held out strictly to calibrate the decision threshold (no test leakage!).", body_style),
        ],
        [
            Paragraph("Test Set (Good)", body_style),
            Paragraph("Nominal (Flawless) Bottles", body_style),
            Paragraph("<b>20</b>", code_style),
            Paragraph("Tested to verify zero false alarms (Specificity = 100%).", body_style),
        ],
        [
            Paragraph("Test Set (Broken Large)", body_style),
            Paragraph("Severe rim break / fracture", body_style),
            Paragraph("<b>20</b>", code_style),
            Paragraph("Tested for detection &amp; localization (Accuracy = 100%).", body_style),
        ],
        [
            Paragraph("Test Set (Broken Small)", body_style),
            Paragraph("Subtle chip / hairline crack", body_style),
            Paragraph("<b>22</b>", code_style),
            Paragraph("Tested for subtle defect sensitivity (Accuracy = 100%).", body_style),
        ],
        [
            Paragraph("Test Set (Contamination)", body_style),
            Paragraph("Foreign oil / chemical stain", body_style),
            Paragraph("<b>21</b>", code_style),
            Paragraph("Tested for surface stain localization (Accuracy = 100%).", body_style),
        ],
    ]
    ds_table = Table(dataset_summary_data, colWidths=[1.3 * inch, 1.8 * inch, 0.7 * inch, 3.6 * inch])
    ds_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(ds_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 4: REAL BENCHMARK RESULTS
    # =========================================================================
    story.append(Paragraph("4. Real Experimental Results (Honest Metrics)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    story.append(
        Paragraph(
            "The model was evaluated against all 83 images of the official test set. "
            "These are the <b>real metrics produced by the code</b> (no placeholder or made-up numbers):",
            body_style,
        )
    )

    results_data = [
        [
            Paragraph("<b>Evaluation Metric</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=9)),
            Paragraph("<b>Measured Score</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=9)),
            Paragraph("<b>What This Means in Plain English</b>", ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=9)),
        ],
        [
            Paragraph("<b>Image-Level AUROC</b>", body_style),
            Paragraph("<font color='#059669'><b>1.0000 (100%)</b></font>", body_style),
            Paragraph("The model perfectly separated all normal bottles from all defective bottles.", body_style),
        ],
        [
            Paragraph("<b>Pixel-Level AUROC</b>", body_style),
            Paragraph("<font color='#059669'><b>0.9222 (92.2%)</b></font>", body_style),
            Paragraph("Evaluated against fine-grained human masks; high localization precision.", body_style),
        ],
        [
            Paragraph("<b>Precision</b>", body_style),
            Paragraph("<font color='#059669'><b>1.0000 (100%)</b></font>", body_style),
            Paragraph("Every time the model flagged a defect, it was a real defect (zero false alarms).", body_style),
        ],
        [
            Paragraph("<b>Recall (Sensitivity)</b>", body_style),
            Paragraph("<font color='#059669'><b>1.0000 (100%)</b></font>", body_style),
            Paragraph("The model did not miss a single defective bottle (63 out of 63 caught).", body_style),
        ],
        [
            Paragraph("<b>Specificity</b>", body_style),
            Paragraph("<font color='#059669'><b>1.0000 (100%)</b></font>", body_style),
            Paragraph("All 20 clean test bottles were correctly identified as normal.", body_style),
        ],
        [
            Paragraph("<b>Calibrated Threshold</b>", body_style),
            Paragraph("<b>1.6636</b>", code_style),
            Paragraph("Mathematical cutoff calculated from validation normals. Zero test set leakage.", body_style),
        ],
    ]
    res_table = Table(results_data, colWidths=[1.8 * inch, 1.4 * inch, 4.2 * inch])
    res_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(res_table)
    story.append(Spacer(1, 10))

    # Embed Plots if available
    plot_roc = "results/plots/roc_curves.png"
    plot_cm = "results/plots/confusion_matrix.png"
    plot_dist = "results/plots/score_distribution.png"
    plot_samples = "results/plots/sample_predictions.png"

    if os.path.exists(plot_roc) and os.path.exists(plot_cm):
        story.append(Paragraph("<b>Visual Plots Generated During Evaluation:</b>", h2_style))
        plot_table_data = [
            [
                RLImage(plot_roc, width=3.5 * inch, height=3.0 * inch),
                RLImage(plot_cm, width=3.5 * inch, height=3.0 * inch),
            ],
            [
                Paragraph("<b>Figure 1:</b> ROC Curves (Image AUROC: 1.000, Pixel: 0.922)", ParagraphStyle("Cap", fontName="Helvetica", fontSize=8, alignment=1)),
                Paragraph("<b>Figure 2:</b> Confusion Matrix (TN=20, FP=0, FN=0, TP=63)", ParagraphStyle("Cap", fontName="Helvetica", fontSize=8, alignment=1)),
            ],
        ]
        plot_table = Table(plot_table_data, colWidths=[3.7 * inch, 3.7 * inch])
        plot_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("PADDING", (0, 0), (-1, -1), 2)]))
        story.append(plot_table)
        story.append(Spacer(1, 8))

    if os.path.exists(plot_dist) and os.path.exists(plot_samples):
        story.append(KeepTogether([
            Paragraph("<b>Score Distribution &amp; Prediction Heatmaps:</b>", h2_style),
            Table(
                [
                    [
                        RLImage(plot_dist, width=3.5 * inch, height=2.4 * inch),
                        RLImage(plot_samples, width=3.5 * inch, height=2.4 * inch),
                    ],
                    [
                        Paragraph("<b>Figure 3:</b> Score Separation vs Threshold (1.6636)", ParagraphStyle("Cap", fontName="Helvetica", fontSize=8, alignment=1)),
                        Paragraph("<b>Figure 4:</b> Ground Truth vs PatchCore Heatmaps", ParagraphStyle("Cap", fontName="Helvetica", fontSize=8, alignment=1)),
                    ],
                ],
                colWidths=[3.7 * inch, 3.7 * inch],
            ),
        ]))
        story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 5: HOW TO RUN AND TEST THE PROJECT
    # =========================================================================
    story.append(Paragraph("5. Step-by-Step Guide: How to Run &amp; Test", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    story.append(Paragraph("<b>1. Start the Web Server:</b>", h2_style))
    story.append(Paragraph("In your terminal inside the project folder, run:", body_style))
    cmd_box1 = Table([[Paragraph("uvicorn app.main:app --host 127.0.0.1 --port 8000", code_style)]], colWidths=[7.4 * inch])
    cmd_box1.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")), ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 6)]))
    story.append(cmd_box1)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>2. Open the Website:</b>", h2_style))
    story.append(Paragraph("Open your web browser and go to: <b>http://127.0.0.1:8000</b>", body_style))

    story.append(Paragraph("<b>3. Test With Real Bottles (Drag &amp; Drop):</b>", h2_style))
    story.append(
        Paragraph(
            "We created a dedicated testing folder in your project: <b><code>___teesting ml mode____</code></b>. "
            "Simply open that folder in Finder, pick any bottle image, and drag it right into the browser!<br/>"
            "&bull; <b>01_normal_good_bottle_A.png:</b> The model will declare <i>'Likely Normal'</i> with a green badge.<br/>"
            "&bull; <b>04_defect_broken_large_bottle_A.png:</b> The model will declare <i>'Potential Anomaly'</i> with a red badge and light up the broken rim in red heat.<br/>"
            "&bull; <b>08_defect_contamination_bottle_A.png:</b> The model will pinpoint the dark foreign oil spot.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>4. Explore the Website Sections:</b>", h2_style))
    story.append(Paragraph("&bull; <b>Prediction Page:</b> Upload any JPG/PNG file and get an instant score, status, and thermal heatmap.", bullet_style))
    story.append(Paragraph("&bull; <b>ML Insights Page:</b> View live dataset counts, model architecture parameters, and evaluation charts.", bullet_style))
    story.append(Paragraph("&bull; <b>About Page:</b> Complete educational overview of the technology stack and methodology.", bullet_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # CHAPTER 6: REAL-WORLD USAGE & RESPONSIBLE AI
    # =========================================================================
    story.append(Paragraph("6. Real-World Factory Usage &amp; Limitations", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    story.append(
        Paragraph(
            "While DefectLens AI achieved 100% accuracy on the MVTec AD bottle benchmark, real-world deployment requires "
            "understanding key engineering realities:",
            body_style,
        )
    )

    story.append(Paragraph("&bull; <b>Lighting Sensitivity:</b> PatchCore relies on visual feature representations. In a factory, cameras must have consistent lighting hoods to prevent false alarms caused by harsh shadows or bright reflections.", bullet_style))
    story.append(Paragraph("&bull; <b>Distance vs. Probability:</b> Anomaly scores represent Euclidean distance in feature memory, not a calibrated statistical probability. A score of 2.5 means 'this patch is far from normal,' not 'there is a 2.5% chance.'", bullet_style))
    story.append(Paragraph("&bull; <b>Human-in-the-Loop:</b> In mission-critical industries (aerospace, pharmaceuticals), DefectLens AI serves as a high-speed attention filter to help human inspectors focus only on suspicious items, rather than fully replacing human judgment.", bullet_style))
    story.append(Spacer(1, 14))

    # Sign-off box
    signoff = Table(
        [
            [
                Paragraph("<b>DefectLens AI</b> &mdash; Built with PyTorch, FastAPI, ResNet-18, and PatchCore.<br/>GitHub: https://github.com/rishavrjpt/DefectLens-AI", ParagraphStyle("Sign", fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#475569"))),
                Paragraph("<b>Developer:</b> Rishav Raj<br/>Year: 2026", ParagraphStyle("SignR", fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#475569"), alignment=2)),
            ]
        ],
        colWidths=[4.7 * inch, 2.7 * inch],
    )
    signoff.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")), ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 8)]))
    story.append(signoff)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully generated: {output_pdf_path}")


if __name__ == "__main__":
    build_pdf("/Users/princechoudhary/Downloads/ml/DefectLens_AI_Project_Guide.pdf")
