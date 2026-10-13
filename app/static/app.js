/**
 * DefectLens AI - Interactive Client Application
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const navButtons = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      navButtons.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetTab);
      if (targetPane) targetPane.classList.add("active");

      if (targetTab === "tab-insights") {
        loadInsights();
      }
    });
  });

  // State
  let selectedFile = null;
  let modelLoaded = false;
  let activeThreshold = 0.5;

  // DOM Elements - Prediction Tab
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const previewMeta = document.getElementById("previewMeta");
  const removeImgBtn = document.getElementById("removeImgBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnText = document.getElementById("btnText");

  const emptyResultState = document.getElementById("emptyResultState");
  const activeResults = document.getElementById("activeResults");
  const statusBanner = document.getElementById("statusBanner");
  const statusBadge = document.getElementById("statusBadge");
  const statusHeadline = document.getElementById("statusHeadline");
  const statusSubline = document.getElementById("statusSubline");
  const decisionTag = document.getElementById("decisionTag");

  const valScore = document.getElementById("valScore");
  const valThreshold = document.getElementById("valThreshold");
  const valRatio = document.getElementById("valRatio");
  const valArea = document.getElementById("valArea");
  const ratioHint = document.getElementById("ratioHint");

  const gaugeBar = document.getElementById("gaugeBar");
  const gaugeLabel = document.getElementById("gaugeLabel");
  const thresholdMarker = document.getElementById("thresholdMarker");

  const resOrigImg = document.getElementById("resOrigImg");
  const resHeatmapImg = document.getElementById("resHeatmapImg");
  const resOverlayImg = document.getElementById("resOverlayImg");
  const visualsGrid = document.getElementById("visualsGrid");
  const viewToggleBtns = document.querySelectorAll(".toggle-btn");

  const statusPillText = document.getElementById("statusPillText");
  const modelStatusPill = document.getElementById("modelStatusPill");
  const systemBanner = document.getElementById("systemBanner");
  const bannerMessage = document.getElementById("bannerMessage");

  // Initial Health Check
  checkHealth();

  async function checkHealth() {
    try {
      const resp = await fetch("/api/health");
      const data = await resp.json();
      modelLoaded = data.model_loaded;

      if (modelLoaded) {
        statusPillText.textContent = `Model Active (${data.category})`;
        const dot = modelStatusPill.querySelector(".status-dot");
        dot.className = "status-dot dot-green";
        systemBanner.style.display = "none";
      } else {
        statusPillText.textContent = "Model Not Trained";
        const dot = modelStatusPill.querySelector(".status-dot");
        dot.className = "status-dot dot-yellow";
        systemBanner.style.display = "block";
        bannerMessage.textContent =
          "No trained model found yet. Run 'python src/train.py' to construct the PatchCore memory bank.";
      }
    } catch (err) {
      statusPillText.textContent = "Server Offline";
      const dot = modelStatusPill.querySelector(".status-dot");
      dot.className = "status-dot dot-red";
    }
  }

  // File Upload Handlers
  browseBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (e.target !== removeImgBtn && !previewContainer.contains(e.target)) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and Drop
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files[0]) {
      handleFileSelected(dt.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.match("image.*")) {
      alert("Please select an image file (PNG or JPEG).");
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzonePrompt.style.display = "none";
      previewContainer.style.display = "flex";
      previewMeta.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  removeImgBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    dropzonePrompt.style.display = "block";
    previewContainer.style.display = "none";
    analyzeBtn.disabled = true;
    emptyResultState.style.display = "block";
    activeResults.style.display = "none";
    decisionTag.textContent = "Awaiting Input";
  });

  // Demo Sample Generator (Allows instant testing with realistic synthetic samples)
  const sampleBtns = document.querySelectorAll(".sample-btn");
  sampleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-type");
      generateSampleImage(type);
    });
  });

  function generateSampleImage(type) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");

    // Dark metallic industrial background
    ctx.fillStyle = "#23262d";
    ctx.fillRect(0, 0, 256, 256);

    // Bottle neck & body
    ctx.fillStyle = "#6e96a0";
    ctx.fillRect(105, 25, 46, 35);

    ctx.fillStyle = "#82b4c3";
    ctx.beginPath();
    ctx.roundRect(70, 60, 116, 165, 18);
    ctx.fill();

    // Specular highlight line
    ctx.strokeStyle = "#d2ebf5";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(95, 70);
    ctx.lineTo(95, 215);
    ctx.stroke();

    if (type === "broken") {
      // Simulate broken jagged edge on bottle side
      ctx.fillStyle = "#23262d";
      ctx.beginPath();
      ctx.moveTo(65, 120);
      ctx.lineTo(98, 140);
      ctx.lineTo(75, 160);
      ctx.lineTo(60, 145);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (type === "contamination") {
      // Simulate dark oil/foreign contaminant spot
      ctx.fillStyle = "#28140f";
      ctx.beginPath();
      ctx.arc(135, 135, 14, 0, Math.PI * 2);
      ctx.fill();
    }

    canvas.toBlob((blob) => {
      const demoFile = new File([blob], `demo_${type}_bottle.png`, { type: "image/png" });
      handleFileSelected(demoFile);
    }, "image/png");
  }

  // Prediction API Execution
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    btnSpinner.style.display = "inline-block";
    btnText.textContent = "Analyzing Patches...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok) {
        const errorDetail = data.detail?.detail || data.detail?.error || "Inference failed";
        const action = data.detail?.suggested_action || "";
        alert(`Error: ${errorDetail}\n${action}`);
        return;
      }

      displayPrediction(data);
    } catch (err) {
      alert(`Network error connecting to DefectLens AI API: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      btnSpinner.style.display = "none";
      btnText.textContent = "Analyze Image";
    }
  });

  function displayPrediction(data) {
    emptyResultState.style.display = "none";
    activeResults.style.display = "block";

    const isAnomaly = data.is_anomaly;
    statusBanner.className = `status-banner ${isAnomaly ? "anomaly" : "normal"}`;
    statusBadge.textContent = data.status;
    decisionTag.textContent = data.status;

    if (isAnomaly) {
      statusHeadline.textContent = "Surface Anomaly Flagged";
      statusSubline.textContent = `Anomaly score (${data.anomaly_score}) exceeds normal baseline (${data.threshold}).`;
      ratioHint.textContent = "Anomaly Detected";
      ratioHint.style.color = "var(--danger)";
    } else {
      statusHeadline.textContent = "Surface Verified Likely Normal";
      statusSubline.textContent = `Feature distances (${data.anomaly_score}) remain within safe tolerance threshold (${data.threshold}).`;
      ratioHint.textContent = "Safe Margin";
      ratioHint.style.color = "var(--success)";
    }

    valScore.textContent = data.anomaly_score.toFixed(3);
    valThreshold.textContent = data.threshold.toFixed(3);
    valRatio.textContent = `${data.score_ratio.toFixed(2)}x`;
    valArea.textContent = `${data.anomalous_area_percentage.toFixed(1)}%`;

    // Visual gauge
    const ratio = Math.min(Math.max(data.score_ratio, 0.0), 2.0);
    const barWidth = Math.min(Math.max((ratio / 2.0) * 100, 4), 100);
    gaugeBar.style.width = `${barWidth}%`;
    gaugeLabel.textContent = `${data.anomaly_score.toFixed(3)} / ${data.threshold.toFixed(3)}`;

    // Set images
    resOrigImg.src = data.original_image_base64;
    resHeatmapImg.src = data.heatmap_base64;
    resOverlayImg.src = data.overlay_base64;

  }

  // View Toggles
  viewToggleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      viewToggleBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const view = btn.getAttribute("data-view");

      if (view === "overlay") {
        visualsGrid.style.gridTemplateColumns = "1fr";
        document.querySelectorAll(".visual-col")[0].style.display = "none";
        document.querySelectorAll(".visual-col")[1].style.display = "none";
        document.querySelectorAll(".visual-col")[2].style.display = "flex";
      } else {
        visualsGrid.style.gridTemplateColumns = "repeat(3, 1fr)";
        document.querySelectorAll(".visual-col").forEach((c) => (c.style.display = "flex"));
      }
    });
  });


  // ML Insights Loader
  async function loadInsights() {
    try {
      const resp = await fetch("/api/insights");
      const data = await resp.json();

      const ds = data.dataset_summary;
      const ts = data.training_summary;
      const em = data.evaluation_metrics;

      // Dataset Summary Card
      if (ds) {
        document.getElementById("insDatasetName").textContent = ds.dataset_name || "MVTec AD";
        document.getElementById("insCategory").textContent = `Category: ${ds.category}`;
        document.getElementById("dtCategory").textContent = ds.category;
        document.getElementById("dtTrainCount").textContent = `${ds.train_good_count} images`;
        document.getElementById("dtTestGoodCount").textContent = `${ds.test_good_count} images`;
        const defectTotal = Object.values(ds.test_defect_counts || {}).reduce((a, b) => a + b, 0);
        document.getElementById("dtTestDefectCount").textContent = `${defectTotal} images`;
        document.getElementById("dtMaskCount").textContent = `${ds.ground_truth_mask_count} masks`;
        document.getElementById("dtResolution").textContent = ds.image_resolution
          ? `${ds.image_resolution[0]}x${ds.image_resolution[1]} px`
          : "Standard MVTec";
        document.getElementById("dtStatus").textContent = ds.status === "verified" ? "Verified" : ds.status;

        const defectRow = document.getElementById("defectTagsRow");
        defectRow.innerHTML = "";
        for (const [k, v] of Object.entries(ds.test_defect_counts || {})) {
          const span = document.createElement("span");
          span.className = "tag";
          span.textContent = `${k} (${v})`;
          defectRow.appendChild(span);
        }
      }

      // Architecture Card
      if (ts) {
        document.getElementById("archBackbone").textContent = `${ts.backbone} (ImageNet Pretrained)`;
        document.getElementById("archLayers").textContent = (ts.layers || []).join(", ");
        document.getElementById("archMemorySize").textContent = `${ts.memory_bank_size} patches`;
        document.getElementById("archDim").textContent = `${ts.feature_dim} channels`;
        document.getElementById("archDevice").textContent = ts.device_used;
        document.getElementById("insThreshold").textContent = ts.calibrated_threshold?.toFixed(4) || "--";
      }

      // Evaluation Metrics Card
      if (em) {
        document.getElementById("insImageAuroc").textContent = em.image_auroc?.toFixed(4) || "--";
        document.getElementById("insAurocSub").textContent = `Evaluated on ${em.test_samples_total} test images`;
        document.getElementById("insPixelAuroc").textContent =
          typeof em.pixel_auroc === "number" ? em.pixel_auroc.toFixed(4) : em.pixel_auroc || "N/A";

        document.getElementById("evalMetricsStrip").style.display = "flex";
        document.getElementById("emF1").textContent = em.f1_score?.toFixed(3) || "--";
        document.getElementById("emPrecision").textContent = em.precision?.toFixed(3) || "--";
        document.getElementById("emRecall").textContent = em.recall?.toFixed(3) || "--";
        document.getElementById("emSpecificity").textContent = em.specificity?.toFixed(3) || "--";
        document.getElementById("emAccuracy").textContent = `${((em.accuracy || 0) * 100).toFixed(1)}%`;

        document.getElementById("evalStatusTag").textContent = "Evaluation Complete";

        // Load plots
        loadPlot("plotRocWrap", "/results/plots/roc_curves.png");
        loadPlot("plotCmWrap", "/results/plots/confusion_matrix.png");
        loadPlot("plotDistWrap", "/results/plots/score_distribution.png");
        loadPlot("plotSamplesWrap", "/results/plots/sample_predictions.png");
      } else {
        document.getElementById("evalStatusTag").textContent = "Not Evaluated Yet";
        const placeholderMsg = `
          <div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.85rem;">
            Run evaluation with: <br/><code class="font-mono" style="background:#e2e8f0; padding:2px 6px; border-radius:4px;">python src/evaluate.py</code>
          </div>`;
        document.getElementById("plotRocWrap").innerHTML = placeholderMsg;
        document.getElementById("plotCmWrap").innerHTML = placeholderMsg;
        document.getElementById("plotDistWrap").innerHTML = placeholderMsg;
        document.getElementById("plotSamplesWrap").innerHTML = placeholderMsg;
      }
    } catch (e) {
      console.error("Failed to load insights:", e);
    }
  }

  function loadPlot(containerId, plotPath) {
    const wrap = document.getElementById(containerId);
    wrap.innerHTML = `<img src="${plotPath}?t=${Date.now()}" alt="Evaluation Plot" onerror="this.parentElement.innerHTML='<span class=\\'placeholder-text\\'>Plot generation pending</span>'" />`;
  }
});
