Optional user instruction:
"Use image files plus metadata.csv to classify image shape. Validate image/metadata pairing before training."

Fault:
- metadata.csv references img_4.png, but img_4.png does not exist.

Expected:
- File grouping/perception should detect image + metadata modality.
- Missing paired image should be reported.
- Error category: missing_artifact / broken_pair.
- Pipeline should either exclude the broken row explicitly and report it, or stop safely; it must not fabricate an image.
