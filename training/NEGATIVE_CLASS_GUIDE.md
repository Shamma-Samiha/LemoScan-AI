# Not lemon leaf class

The disease model is a closed-set classifier unless it has a negative class. To stop human, fern, mango leaf, table, soil, or blurry images from being labeled as lemon diseases, add real negative examples to:

- `dataset/train/Not lemon leaf`
- `dataset/val/Not lemon leaf`
- `dataset/test/Not lemon leaf`

Recommended minimum for a thesis-stage model:

- Train: 300-800 images
- Validation: 60-150 images
- Test: 60-150 images

Include varied negatives:

- Human faces, hands, and clothing
- Ferns and ornamental leaves
- Other crop leaves
- Soil, pots, tools, paper, sky, table backgrounds
- Blurry or very dark images
- Multiple leaves or non-single-leaf scenes that should be rejected

After adding images, run:

```powershell
python training\train_inceptionv3.py
```

The script saves:

- `model/inceptionv3_best_model.keras`
- `training/inceptionv3_class_indices.json`
- `model/model_metadata.json`

The app reads `model/model_metadata.json` automatically, so once training finishes it will use the InceptionV3 model, `299x299` input size, and InceptionV3 preprocessing.
