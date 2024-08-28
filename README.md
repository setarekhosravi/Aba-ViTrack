# Aba-ViTrack

The official implementation of the ICCV 2023 paper [Adaptive and Background-Aware Vision Transformer for Real-Time UAV Tracking](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Adaptive_and_Background-Aware_Vision_Transformer_for_Real-Time_UAV_Tracking_ICCV_2023_paper.pdf)

<p align="center">
  <img width="85%" src="https://github.com/xyyang317/Aba-ViTrack/blob/main/arch.png" alt="Framework"/>
</p>

## Install the environment
This code has been tested on Ubuntu 18.04, CUDA 10.2. Please install related libraries before running this code:
   ```
   conda create -n abavitrack python=3.8
   conda activate abavitrack
   bash install.sh
   ```

## Model and raw results
The trained model and the raw tracking results are provided in the [Baidu Netdisk](https://pan.baidu.com/s/13aXfsihrbrh8WMu6XYTthA?pwd=nen9)(code: nen9) or [Google Drive](https://drive.google.com/drive/folders/17FYC5xl8EaBL21Zbhj7yQcU0lg9mblwx?usp=drive_link).

## Run demo
Download the model and put it in checkpoints

   ```
   python demo.py --initial_bbox 499 421 102 179 
   ```

## Run Aba-ViTrack on custom video

### Update tracker using ground truth

```bash
python run_eval.py --data_path /path/to/video/file --save_path /path/to/saving/dir --update_rate 10
```

### Update tracker using YOLO as detector

```bash
python track_by_detect.py --data_path /path/to/video/file --save_path /path/to/saving/dir --update_rate 10 --det_weights /path/to/yolo/weights
```

> 📝 By running these codes, you can run your tracker on a custom video and store the results in xyxy format. These codes are used to evaluate your tracker using detection methods to find its effect on detection performance.

## Citation
```
@InProceedings{Li_2023_ICCV,
    author    = {Li, Shuiwang and Yang, Yangxiang and Zeng, Dan and Wang, Xucheng},
    title     = {Adaptive and Background-Aware Vision Transformer for Real-Time UAV Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13989-14000}
}
```
