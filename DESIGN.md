# attnvis — Design Document

## 1. Overview

**attnvis** is a lightweight framework for **visualising and analysing attention mechanisms in vision transformers**, with a focus on **understanding, debugging, and teaching model behaviour**.

The project is intentionally staged:

- **Phase 1a:** Vision Transformer (ViT) — global self-attention, patch-based semantics  
- **Phase 1b:** Swin Transformer — hierarchical, window-based attention  
- **Phase 2:** DETR — encoder–decoder attention for object detection

Each phase builds on the previous one, increasing **architectural complexity** while reusing the same core abstractions.

The emphasis is on **correct semantics**, **clear abstractions**, and **reproducible analysis**, not UI polish.

---

## 2. Problem Statement

Attention is central to modern vision models, yet:

- Its semantics differ significantly across architectures
- Visualisations are often misleading or oversimplified
- Tooling is fragmented and notebook-bound

In particular:

- **ViT:** global self-attention over image patches  
- **Swin:** local window attention with hierarchical feature maps  
- **DETR:** non-spatial object queries and encoder–decoder cross-attention  

Most existing tools:
- Conflate these meanings
- Visualise attention without stating *what it represents*
- Do not scale across architectural families

**attnvis** aims to provide **architecture-aware attention visualisation** with explicit assumptions and boundaries.

---

## 3. Goals & Non-Goals

### Goals
- Extract and visualise attention **with correct architectural semantics**
- Support:
  - ViT global self-attention
  - Swin window-based and shifted-window attention
  - DETR encoder self-attention and decoder cross-attention
- Enable layer-wise and head-wise comparison
- Be CLI-driven and reproducible
- Serve as a learning and debugging tool

### Non-Goals
- Training or fine-tuning models
- Supporting all transformer variants
- Providing theoretical guarantees of interpretability
- Building a full interactive dashboard

---

## 4. High-Level Architecture

       +-------------+
       |     CLI     |
       |  (typer)    |
       +------+------+ 
              |
              v
    +---------+----------+
    |  Model Adapter     |  <-- ViT / Swin / DETR
    +---------+----------+
              |
              v
    +---------+----------+
    | Attention Extractor|
    | (arch-specific)   |
    +---------+----------+
              |
              v
    +---------+----------+
    | Visualisation Core |
    | (shared logic)    |
    +---------+----------+
              |
              v
          Outputs



**Key design principle:**  
> Extraction is architecture-specific, visualisation logic is shared.

---

## 5. Core Concepts

### Attention Semantics (Explicit)

| Model | Attention Type | Meaning |
|------|----------------|--------|
| ViT | Self-attention | Global patch ↔ patch relationships |
| Swin | Window self-attention | Local patch relationships within windows |
| Swin (shifted) | Shifted window attention | Cross-window information flow |
| DETR Encoder | Self-attention | Contextualised image features |
| DETR Decoder | Cross-attention | Object queries attending to image regions |

Each visualisation explicitly states:
- which tokens attend
- to what representation
- at what spatial scale
- at what stage in the pipeline

---

## 6. Phase 1a — Vision Transformer (ViT)

### Why ViT First
- Simple spatial token-to-patch mapping
- Single attention mechanism
- Ideal for validating visualisation correctness

### Capabilities
- Extract per-layer, per-head attention
- Visualise:
  - CLS → patch attention
  - patch-level heatmaps
  - all-head grids per layer
- Attention rollout across layers

### Success Criteria
- Correct spatial alignment
- Distinct head behaviours
- Reproducible CLI outputs

---

## 7. Phase 1b — Swin Transformer (Hierarchical Attention)

### Motivation

Swin introduces **hierarchy and locality**, bridging the gap between ViT and detection models:

- Window-based self-attention
- Shifted windows for cross-window interaction
- Multi-stage feature pyramid

Correctly visualising Swin attention demonstrates **understanding of inductive bias**, not just attention math.

---

### Swin Components to Visualise

#### 1. Window-Based Self-Attention
- Attention confined to fixed-size windows
- Patch-to-patch relationships within windows

#### 2. Shifted Window Attention
- How attention crosses window boundaries
- Comparison: non-shifted vs shifted layers

#### 3. Hierarchical Stages
- Attention at different spatial resolutions
- Effect of patch merging on attention granularity

---

### Visualisation Principles for Swin

- Attention is **local by design**
- Visualisations must respect window boundaries
- Shifted-window attention is visualised relative to the *unshifted image grid*
- No artificial global attention aggregation

---

### Example Outputs

- Attention heatmaps per window
- Overlay showing window boundaries
- Side-by-side comparison:
  - regular window vs shifted window
- Attention evolution across Swin stages

---

## 8. Phase 2 — DETR (System-Level Extension)

### Motivation

DETR introduces:
- encoder–decoder attention
- learned object queries
- non-spatial attention semantics

This phase demonstrates **system-level reasoning** across backbone, transformer, and prediction heads.

---

### DETR Components to Visualise

#### 1. Backbone Feature Maps
- Spatial resolution and channel depth
- Mapping to encoder tokens

#### 2. Encoder Self-Attention
- Feature token ↔ feature token
- Global context formation

#### 3. Decoder Cross-Attention (Core Focus)
- Object queries → image features
- Query-to-region alignment
- Query specialisation across layers

#### 4. Decoder Self-Attention (Optional)
- Query ↔ query interactions
- Redundancy and diversity between queries

---

### Visualisation Principles for DETR

- Object queries are **not spatial**
- Cross-attention is mapped onto image space
- Encoder and decoder attention are never mixed
- Matching is treated as post-hoc analysis, not attention

---

## 9. Staged Implementation Plan

### Stage 0 — Project Skeleton
- Repo layout
- Docker (CPU-only)
- CLI entrypoint

### Stage 1 — ViT Support
- Model loading
- Attention extraction
- Patch-level visualisation

### Stage 2 — Swin Support
- Window-aware attention extraction
- Shifted window handling
- Hierarchical attention visualisation

### Stage 3 — DETR Adapter
- Backbone / encoder / decoder separation
- Attention hooks
- Query indexing

### Stage 4 — DETR Visualisation
- Query-to-image attention
- Layer-wise query evolution
- Failure case analysis

---

## 10. Repo Structure
```markdown

attnvis/
├── attnvis/
│ ├── core/
│ │ ├── viz.py
│ │ └── utils.py
│ ├── vit/
│ │ ├── model.py
│ │ ├── extract.py
│ │ └── viz.py
│ ├── swin/
│ │ ├── model.py
│ │ ├── extract.py
│ │ └── viz.py
│ ├── detr/
│ │ ├── backbone.py
│ │ ├── encoder.py
│ │ ├── decoder.py
│ │ └── viz.py
│ ├── main.py
│ └── cli.py
├── examples/
│ ├── vit/
│ ├── swin/
│ └── detr/
├── tests/
├── DESIGN.md
└── README.md

```


---

## 11. Testing Strategy

- Shape and consistency checks for attention tensors
- Window-boundary correctness tests for Swin
- Smoke tests for CLI
- Optional visual regression tests for example outputs

---

## 12. Success Criteria

The project succeeds if:

- Attention semantics are **explicit and correct**
- ViT, Swin, and DETR share abstractions but differ in interpretation
- A reviewer can reason about model behaviour from outputs alone
- The codebase is extensible to new transformer variants

---

## 13. Out of Scope (Future Work)

- Deformable DETR
- Video transformers
- Interactive UI
- Training-time attention monitoring




