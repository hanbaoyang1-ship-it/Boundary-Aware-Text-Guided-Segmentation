# Data preparation

Datasets are not redistributed in this repository.

- QaTa-COV19: <https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset>
- MosMedData: <https://mosmed.ai/datasets/covid19_1110>

The manuscript uses the structured text descriptions and fixed partitions from the LViT benchmark protocol. Obtain those annotations from the original benchmark source and respect the dataset licenses.

## Expected metadata

Each split is a CSV, TSV, XLS, or XLSX table. The recommended columns are:

| image | text | mask (optional) |
|---|---|---|
| `slice_0001.png` | `Bilateral pulmonary infection...` | `slice_0001_mask.png` |

If `mask_column` is `null`, the loader expects image and mask files to have the same filename. If named `image` and `text` columns are absent, the loader falls back to the first two columns for compatibility with the supplied MosMed code.

Example layout:

```text
data/
├── MosMedData/
│   ├── frames/
│   ├── masks/
│   ├── Train_text_MosMedData.xlsx
│   ├── Val_text_MosMedData.xlsx
│   └── Test_text_MosMedData.xlsx
└── QaTa-COV19/
    ├── images/
    ├── masks/
    ├── train.xlsx
    ├── val.xlsx
    └── test.xlsx
```

Edit the paths and column names in `configs/*.yaml` if your benchmark copy uses a different layout. The original attachment did not include the QaTa metadata filenames, so the QaTa names above are explicit placeholders rather than a claim about the original experiment filesystem.
