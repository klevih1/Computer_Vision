# Reliability-Aware Wildlife Detection in Aerial Drone Imagery

We detect animals in thermal drone footage and, for each detection, also say how much
you can trust it: is the animal blurry, and is it partly hidden? A normal detector just
draws a box. The point here is to tell a researcher which boxes are solid and which ones
to double-check.

## What it does

Three models working together:

1. **Detection** (YOLOv11) finds the animals.
2. **Sharpness classifier** marks each one as sharp or blurry.
3. **Occlusion classifier** marks each one as visible or occluded.

Plus a pipeline that ties them together: an image goes in, and every detected animal
comes out with a species label and a reliability tag (reliable / caution / unreliable).

## Dataset

The ALFS subset of the BAMBI project: thermal aerial drone images in YOLO format. The
animals are tiny in the frame, often motion-blurred, and frequently half-hidden behind
terrain. The frames come from video, so we split the data by whole flight (not by
single frame) to keep near-identical frames from landing in both train and test.

After cleaning:
- 17,630 images, 55,386 boxes, 190 flights
- 7 species: alpine-ibex, chamois, dog, fallow-deer, human, red-deer, roe-deer

## Two runs: what we fixed and why it got better

We trained the detector, looked closely at the results, and trained it again. The first
run scored low, and digging into it turned up a few real problems. After fixing them and
re-running, every detection metric went up:

| Problem in the first run | Fix |
|--------------------------|-----|
| The model trained at high resolution (1280) but was evaluated at 640 - the tiny animals basically vanish at the smaller size, so the score looked worse than the model really was | Evaluate at the same 1280 it was trained on |
| The animal crops were saved as JPEG, which blurs them - a problem when the whole point is measuring blur | Save crops as lossless PNG |
| "no-animal" was being used as a detection class - you can't draw a box around nothing, and it dragged the average down | Drop it, keep the 7 real species |

Detection results on the test set, first run vs after the fixes:

| Metric | First run | After fixes |
|--------|-----------|-------------|
| mAP@0.5 | 0.185 | **0.217** |
| mAP@0.5:0.95 | 0.063 | **0.080** |
| Precision | 0.308 | **0.512** |
| Recall | 0.204 | **0.221** |

The same crop fix (PNG instead of JPEG) also helped the sharpness classifier, whose
F1 went from 0.91 to 0.96.

## How it works

**Data prep (Notebooks 1 and 2).** We validate the annotations, drop duplicate images,
split by flight (~70/15/15), and cut out a small padded crop of every animal.

**Reliability labels.**
- *Sharpness* is scored with Laplacian variance (how much fine detail a crop has), with
  the dark background masked out. A threshold splits crops into sharp vs blurry; we
  picked it by eye from a calibration grid.
- *Occlusion* is read straight from the BAMBI class code, whose last digit already records
  whether the original annotators saw the animal as visible or occluded - so no manual
  labelling needed.

**Models (Notebook 3).** YOLOv11s (pretrained on COCO) for detection, and two
EfficientNet-B0 networks for the sharpness and occlusion classifiers. Both classifiers
use class weights because the data is heavily imbalanced.

**Pipeline.** One function: image -> detect -> crop each animal -> run both classifiers ->
return each detection tagged reliable / caution / unreliable.

## Results (test set)

**Detection, per class (mAP@0.5):**

| Species | mAP@0.5 |
|---------|---------|
| red-deer | 0.427 |
| alpine-ibex | 0.317 |
| fallow-deer | 0.275 |
| human | 0.227 |
| roe-deer | 0.181 |
| dog | 0.063 |
| chamois | 0.030 |

red-deer (by far the most common class) is detected best; the rarest species are hardest.

**Sharpness classifier:** F1 (macro) 0.957, ROC-AUC 0.993
**Occlusion classifier:** F1 (macro) 0.616, ROC-AUC 0.670

**Manual sharpness check (Notebook 4):** a classifier trained on 100 hand-picked images
(50 clear, 50 blurry) reached F1 0.83 on the held-out images. Worth noting: the
auto-labelled sharpness classifier scores higher (0.96 vs 0.83) - but that's partly
because it's effectively learning to reproduce the Laplacian rule it was labelled with,
while this manual version is the stricter, human-grounded check.

A visual walkthrough of all the charts and tables is in [SHOWCASE.md](SHOWCASE.md).

## Notebooks

| File | What it does |
|------|--------------|
| `Notebook_1.ipynb` | Load and clean the data, save the annotation table |
| `Notebook_2.ipynb` | Split by flight, make the crops, build the reliability labels |
| `Notebook_3.ipynb` | Train the detector + classifiers, evaluate, run the pipeline |
| `Notebook_4.ipynb` | Manual sharpness classifier (100 hand-picked images) |
| `reports/` | Charts and metric JSONs |
| `notebook4_outputs/` | Notebook 4's charts and metrics |

The trained model weights are included in `models/` (YOLO detector + the two
classifiers + the manual-sharpness model). The raw dataset, crops and CSVs are not in
the repo (too big) - the dataset is the BAMBI ALFS thermal subset; place it in a
`dataset/` folder next to the notebooks and run Notebooks 1 and 2 to regenerate the
crops and split.

## Running it

Relative paths, so keep `dataset/` next to the notebooks.

```bash
pip install ultralytics opencv-python pandas numpy matplotlib seaborn pyyaml scikit-learn
```

Run Notebooks 1 -> 2 -> 3 in order (Notebook 3's training wants a GPU). Notebook 4 is
standalone and uses its own set of images.

## A note on how it was run

The heavy training (the YOLO detector and the two classifiers) was done on a cloud GPU (Azure) as detached background scripts, since the laptop we worked on has no GPU and
training is slow. The notebooks then load the saved weights (in `models/`) to evaluate
the models and run the pipeline - so you can reproduce all the results without retraining.

Because training ran as scripts rather than inside the notebooks, a couple of in-notebook
outputs aren't present: the classifier training cells show no live output (we added a
note in those cells), and the per-epoch training curves for the two classifiers aren't plotted (the scripts didn't log them). The confusion matrices, final metrics, and the
YOLO training curves are all there - those are what the results rest on.

## Limits and what's next

- **Class imbalance**: red-deer dominates, so the rare species are detected and classified
  less reliably. Oversampling or stronger class weights would help.
- **Compute**: high-res detection training is slow; an even higher resolution might push
  the small-animal detection further.
- **Sharpness labels** are auto-generated, so that classifier is partly learning to copy a
  formula - Notebook 4's hand-labelled version is the stricter check.
- **Occlusion labels** come from the dataset's own codes

