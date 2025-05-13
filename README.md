## 📄 YoloAug

This repository display the effects of ultralytics augmentation with gradio.

## 📦 Run image transformation

### Run with the current environment
Run the following command to start the Gradio web app in your current environment:
```shell
python ShowImg.py
```
Go to http://127.0.0.1:7860

### Run with Docker
If you prefer to run the Gradio web app in a Docker container, follow these steps:

1. **Pull the Docker image**:
    ```shell
    docker pull ultralytics/ultralytics:8.3.133
    ```

2. **Run the Docker container**:
    ```shell
    docker run -p 7860:7860 -v $(pwd):/app ultralytics/ultralytics:8.3.133
    ```

3. **Go to http://127.0.0.1:7860**

## 📦 Run image Mosaic

### Run with the current environment
Run the following command to start the Gradio web app in your current environment:
```shell
python ShowMosaic.py
```

### Run with Docker
If you prefer to run the Gradio web app in a Docker container, follow these steps:
1. **Pull the Docker image**:
    ```shell
    docker pull ultralytics/ultralytics:8.3.133
    ```
2. **Run the Docker container**:
    ```shell
    docker run -p 7860:7860 -v $(pwd):/app ultralytics/ultralytics:8.3.133
    ```
3. **Go to http://127.0.0.1:7860**

## 📜 License
Ultralytics offers two licensing options to suit different needs:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license) open-source license is perfect for students, researchers, and enthusiasts. It encourages open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for full details.
- **Ultralytics Enterprise License**: Designed for commercial use, this license allows for the seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. If your use case involves commercial deployment, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📜 Notice 
The code is cloned from https://github.com/ultralytics/ultralytics
Original author: Ultralytics
