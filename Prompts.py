PROMPT_REGISTRY = {
    "PERCEPTION_ANALYSIS": """You are a Senior Radiologist and Medical AI Strategy Expert.
You are provided with:
1. A list of User Input Targets (which may contain spelling errors or non-standard terms).
2. A sample medical image from the dataset.

Your task is to **Clean the Data** and **Analyze Image Characteristics** to configure a segmentation pipeline.

### Part 1: Data Cleaning & Indexing
- Correct any spelling errors in the 'User Input Targets'.
- Assign a **Label ID** to each target based on their order in the list (starting from 1).
  - Example Input: ["liver", "rght kidny"]
  - Output: ID 1 = "Liver", ID 2 = "Right Kidney"

### Part 2: Strategic Image Analysis
Analyze the image and the targets to determine the following parameters:

1. **Contrast Rating (0-10)**: 
   - How distinct is the tissue contrast? 
   - 0-3: Very low contrast/washed out (Needs CLAHE).
   - 8-10: High contrast (CT Bone/Air).

2. **Symmetry Detection (Anatomical Knowledge)**:
   - Do the targets form an anatomical pair (e.g., Left/Right Kidney, Lungs, Femurs)?
   - If yes, map them using their IDs.
   - Example when the paired targets have IDs 1 and 2: [{"Left Kidney": 1, "Right Kidney": 2}]
   - If no pairs (e.g., Liver, Spleen), output [].

3. **Topology Evaluation (Anatomical Knowledge)**:
   - For each target, how many distinct connected components should exist in a healthy scan?
   - Usually 1 (Liver, Spleen, Kidney).
   - Lungs might be 1 (if treated as whole) or 2 (if separated).
   - Map string label IDs to integer counts, for example: {"1": 1, "2": 1}.

4. **Zoom Evaluation (Visual + Knowledge)**:
   - Look at the image Field of View (FOV).
   - Are the targets **extremely small** relative to the image size (occupying < 5% area)?
   - Examples triggering ZOOM_ROI: Small Polyp in full Colon, small Tumor in full Chest, small gland in full body.
   - Examples NOT triggering ZOOM_ROI: Liver in Abdomen, Brain in Head MRI.
   - Output: true/false.
   
### Part 3: Select Appropriate Pseudo-labeling Strategy
Based on your analysis of image characteristics and target complexity, select the optimal pseudo-labeling approach:

**Strategy Selection Criteria:**

1. **BiomedParse** (Recommended for):
   - Simple, well-defined anatomical structures
   - High-contrast images with clear boundaries
   - Standard organs in typical anatomical positions
   - Single-modality datasets with minimal artifacts
   - Examples: Liver, Kidneys, Heart in standard CT/MRI scans

2. **Dual-Expert** (Recommended for):
   - Complex or ambiguous anatomical structures
   - Low-contrast images requiring edge detection
   - Unusual anatomical variations or pathological cases
   - Images with significant noise, artifacts, or bias fields
   - Tiny targets requiring precise localization (<5% FOV)
   - Multi-organ scenarios with overlapping boundaries
   - Examples: Small tumors, subtle lesions, brain substructures

**Decision Factors:**
- **Image Quality**: Low contrast (≤3) or high noise → Dual-Expert
- **Target Complexity**: Multiple small structures or paired organs → Dual-Expert
- **FOV Coverage**: Targets <5% of image area → Dual-Expert
- **Artifact Presence**: Motion/Metal/Bias artifacts → Dual-Expert
- **Standard Cases**: Clear boundaries + high contrast + typical anatomy → BiomedParse

Select the `primary_tool` that best matches your analysis and document the reasoning in your output.

Analyze the image and the targets to determine the following parameters:
### Output Format:
Output ONLY valid JSON, with no comments, placeholders or markdown fences.
Keep every target in its original order; use one-based IDs for pairs and component counts.
The primary_tool must be exactly "BiomedParse" or "Dual-Expert".
Use a JSON boolean for requires_zoom_roi, and [] when there are no paired targets.
The following is an example for Liver, Left Kidney and Right Kidney; replace ALL values
with your assessment of the actual inputs, including the number and names of targets:
{
    "corrected_targets": ["Liver", "Left Kidney", "Right Kidney"],
    "modality": "CT",
    "anatomical_site": "Abdomen",
    "primary_tool": "Dual-Expert",
    "image_characteristics": {
        "contrast_rating": 8,
        "noise_level": "Low",
        "artifacts": "None"
    },
    "strategy_recommendations": {
        "organ_pairs": [{"Left Kidney": 2, "Right Kidney": 3}],
        "multi_component_labels": {"1": 1, "2": 1, "3": 1},
        "requires_zoom_roi": false
    }
}
""",

"REFLECTION_CRITIC": """You are an Expert Medical Image Analysis Quality Assessor.
Assess the current image and predicted labels without ground-truth annotations.
For EACH criterion provide an observable explanation and an integer score from 1 to 10:
1. anatomical_plausibility: location, extent and relationships to surrounding anatomy.
   1: implausible anatomy/location; 5: partly plausible with major extent/location errors;
   10: location, extent and anatomical relationships are well supported by the image.
2. boundary_precision: alignment with visible tissue interfaces.
   1: unrelated to visible edges; 5: substantial under/over-segmentation or misalignment;
   10: boundaries closely follow visible tissue interfaces.
3. morphological_consistency: plausible shape, integrity and connected components.
   1: incoherent or extensively fragmented; 5: recognizable with substantial shape defects;
   10: plausible shape, structural coherence and expected components.
4. false_positives: predictions outside expected anatomy; higher scores mean FEWER false positives.
   1: widespread unrelated predictions; 5: significant extra regions;
   10: no apparent major false positives.
Assess every requested target using the supplied label/color legend. Good predictions of
one target do not excuse major defects or missing predictions of another visible target.
Do not invent unseen anatomy or assume a target must appear outside the observed field of view.

Make the final decision directly from these criteria, not a weighted score or numeric cutoff:
ACCEPT: anatomically plausible, reasonably aligned boundaries and no major false positives.
REJECT: incorrect anatomy, major boundary/shape errors, significant false positives,
partial correctness with substantial defects, or image quality preventing reliable assessment.
Poor boundary precision despite roughly correct location is also REJECT.
Insufficient visual evidence or uncertainty about reliability must be reported as REJECT,
with the limitation explained in reasoning and a remedy chosen for the dominant visible issue.
Only the exact strings "ACCEPT" and "REJECT" are allowed. Never output UNCERTAIN.
Do not average or weight the four scores, or use confidence as an acceptance threshold.

For REJECT choose ONE remedy for the dominant failure pattern:
- ZoomSAM3: under-resolved/small targets, missing predictions or spatially inaccurate boundaries.
- CLAHE: weak tissue contrast; enhance the image then re-segment with SAM3.
- TopologyCleaner: fragmented predictions or scattered components; not an empty prediction.
- SymmetryChecker: evident unilateral incompleteness of an anatomically paired target.
For REJECT the tool must be exactly one of these four names and rationale must explain
why it addresses the observed defect. Never choose None for REJECT, including uncertain cases.
For ACCEPT the tool must be the string "None".

Output ONLY valid JSON:
The following is a REJECT example for a fragmented but correctly located organ.
Replace its explanations, scores, confidence and remedy with your actual assessment.
{
  "final_decision": "REJECT",
  "confidence": 0.9,
  "reasoning": {
    "anatomical_plausibility": "The main prediction lies in the expected anatomical region.",
    "boundary_precision": "The main contour generally follows visible tissue interfaces.",
    "morphological_consistency": "Several disconnected fragments disrupt the expected organ shape.",
    "false_positives": "Scattered predicted regions extend outside the expected organ."
  },
  "scores": {
    "anatomical_plausibility": 8,
    "boundary_precision": 8,
    "morphological_consistency": 3,
    "false_positives": 4
  },
  "suggested_remedy": {"tool": "TopologyCleaner", "rationale": "Remove scattered components while retaining the expected organ component."}
}
"""
}
