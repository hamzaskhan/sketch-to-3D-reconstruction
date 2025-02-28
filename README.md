# sketch-to-3D-reconstruction
A Python-based framework for converting 2D sketches and grayscale images into realistic 3D objects. Combines Pix2Pix, Real-ESRGAN, DeOldify, and ShapE models to enhance image quality, colorize, and generate 3D point clouds for detailed reconstructions.

---

This repository implements a framework for converting 2D sketches or grayscale images into 3D objects. The system combines various models, including Pix2Pix, Real-ESRGAN, DeOldify, and ShapE, to handle tasks such as grayscale-to-RGB conversion, resolution enhancement, colorization, and 3D point cloud generation.

### **Overview**

The project aims to improve the process of converting 2D sketches into 3D objects by addressing challenges in texture realism, depth perception, and model generalization. It integrates multiple advanced techniques:
1. **Pix2Pix** (GAN-based) for conditional image generation from sketches.
2. **Real-ESRGAN** for enhancing resolution of generated images.
3. **DeOldify** for colorization of grayscale images.
4. **ShapE** for generating 3D point clouds and converting 2D images to 3D shapes.

This framework is a step forward in improving the real-world applicability of sketch-to-3D tasks, especially in industries such as gaming, animation, and virtual prototyping.

---

### **Features**

- **Sketch-to-3D Pipeline:** Convert sketches to detailed 3D objects with depth and texture improvements.
- **Modular Design:** Supports adding or replacing individual stages of the pipeline, such as image enhancement or 3D point cloud generation.
- **Custom Training:** Models can be trained on custom datasets, or the existing models can be fine-tuned for specific applications.

---

### **Dependencies**

This project is implemented in Python and requires several key libraries. Install them using the following:

```bash
pip install -r requirements.txt
```

The required libraries include:

- `torch` (for deep learning models)
- `torchvision` (for image processing)
- `numpy` (for numerical operations)
- `opencv-python` (for image handling)
- `PIL` (for image processing)
- `Real-ESRGAN` (for resolution enhancement)
- `DeOldify` (for colorization)
- `ShapE` (for 3D point cloud generation)
- `midas` (for depth estimation)

For other dependencies, refer to the `requirements.txt` file.

---

### **Installation**

1. Clone this repository:

```bash
git clone https://github.com/hamzaskhaan/sketch-to-3D-reconstruction.git
cd sketch-to-3D-reconstruction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### **Training the custom Models**

To train **Pix2Pix** on your own dataset, use the following command:

Ensure that the dataset is prepared in a paired format (e.g., sketches paired with corresponding colorized images).

---

### **Contributing**

We welcome contributions to improve the functionality of this framework. If you have suggestions or bug fixes, please fork the repository and submit a pull request. Ensure you follow the contribution guidelines in the `CONTRIBUTING.md` file.

---

### **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Acknowledgments**

- **Pix2Pix**: Isola, P., et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR.
- **Real-ESRGAN**: Wang, X., et al. (2021). Real-ESRGAN: Training Real-World Super-Resolution with Realistic Examples. ECCV.
- **DeOldify**: Antic, J. (2020). DeOldify: Bringing Color to the Past. GitHub.
- **ShapE**: Saharia, C., et al. (2023). ShapE: Generative Shape-Aware Image-to-Image Translation. NeurIPS.
- **Midas**




