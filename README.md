# Mask Matching - ISIC2018

## 1. Introduction

This side-project offers a compact visual tool for comparing a reference segmentation mask with a swarm of candidate mask (with ISIC2018 Dataset)  
It scans a folder full of segmentation PNGs, measures the absolute pixel-sum difference between each candidate and your reference mask, sorts them by similarity, then opens an interactive viewer where you can **slide through all candidates** and see:

- the reference mask
- the candidate mask
- the corresponding RGB image
- the computed matching error

It answers the simple “_Which mask resembles mine the most?_” question with a neatly wrapped visual interface.

---

## 2. Environment

You can build the environment using either **conda** or **pip**; both approaches use the same dependencies listed in `requirements.txt`.

### **requirements.txt**

```

numpy
pillow
tqdm
matplotlib

```

### **Using conda**

```sh
conda create -n maskmatch python=3.10 -y
conda activate maskmatch
pip install -r requirements.txt
```

### **Using pip (no conda)**

```sh
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. How to Run

The script accepts three optional arguments:

| Argument            | Meaning                                              |
| ------------------- | ---------------------------------------------------- |
| `--input_mask_path` | The reference mask used for comparison               |
| `--mask_folder`     | Folder containing candidate mask PNG files           |
| `--img_folder`      | Folder containing the RGB images matched by filename |

Run the helper to see the usage guide:

```sh
python match.py --help
```

A typical run:

```sh
python match.py \
    --input_mask_path ./mask.png \
    --mask_folder ./ISIC2018_Task1_Validation_GroundTruth \
    --img_folder ./ISIC2018_Task1-2_Validation_Input
```

Once executed, the script prints ranking results in the terminal and opens the interactive visualization window with the slider.

---

## 4. Examples

Below are three illustrations demonstrating how the script behaves from start to finish.

---

### **(1) Help Display**

![Help Screenshot](images/image_help.png)

This screenshot shows the console output after running `python match.py --help`.
You’ll see the list of arguments, each accompanied by a short description explaining what it controls.
It provides a quick snapshot of how to steer the script without guessing.

---

### **(2) Matching Plot With Slider**

![Slider UI](images/image_slider.png)

This captures the interactive viewer once the script begins its visual comparison journey.
From left to right you see:

1. your reference mask,
2. the current candidate mask,
3. the corresponding RGB image.

Above them, the figure title highlights the candidate index and the associated error metric.
Below the trio, a slider lets you sweep through all candidates effortlessly — like scrubbing through frames of an anatomical film reel.

---

### **(3) Terminal Output After Running**

![Terminal Output](images/image_terminal.png)

This terminal excerpt appears after the scanning and sorting phase.
It displays:

- the progress bar used during mask comparison
- the top 3 best-matching candidates (sorted by smallest error)
- a confirmation message before launching the visualization window

It provides a clear, concise summary before the visual deep dive begins.
