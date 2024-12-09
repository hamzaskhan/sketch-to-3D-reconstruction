# Install required libraries
!pip install torch torchvision numpy matplotlib pillow numpy-stl opencv-python yolov5

# Clone the MiDaS repository
!git clone https://github.com/isl-org/MiDaS.git

import numpy as np
from PIL import Image
from stl import mesh
from scipy.ndimage import gaussian_filter
import cv2
import torch
from yolov5 import detect  # YOLOv5 for object detection
from torchvision import transforms
from google.colab import files  # For file upload/download in Colab

# YOLOv5
def object_detection_with_yolo(image_path):
    """
    Detects objects in an image using YOLOv5.
    Returns object masks.
    """
    results = detect.run(weights="yolov5s.pt", source=image_path, save_crop=True)
    return results

def depth_estimation_with_dpt(image_path):
    """
    Generates a depth map using the DPT-Hybrid model.
    """
    model_type = "DPT_Hybrid"  # Choose DPT-Hybrid for balanced performance
    model = torch.hub.load("isl-org/MiDaS", model_type)
    model.eval()

    #Transformation
    transform = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform

    #Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((384, 384))  # Resize to ensure compatibility
    img_np = np.array(img).astype(np.float32)

    #transformations
    input_batch = transform(img_np).unsqueeze(0)

    #Squeeze extra dimension
    if input_batch.dim() == 5:  # Check for 5D input
        input_batch = input_batch.squeeze(0)

    # Debug: Check input shape
    print(f"Input batch shape after squeeze: {input_batch.shape}")  # Should be [1, 3, 384, 384]

    with torch.no_grad():
        depth_map = model(input_batch).squeeze().cpu().numpy()

    return depth_map



#Depth Normalization and Sharpening
def refine_depth_map(depth_map):
    """
    Normalizes and sharpens the depth map using Sobel filtering and Gaussian smoothing.
    """
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.hypot(sobel_x, sobel_y)

    enhanced_depth = depth_map + edges * 0.1
    smoothed_depth = gaussian_filter(enhanced_depth, sigma=2.0)

    return smoothed_depth

#Depth-to-Normal Conversion
def depth_to_normal(depth_map):
    """
    Converts a depth map to surface normals for geometric refinement.
    """
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    grad_z = np.ones_like(depth_map)

    norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    normals = np.stack((grad_x / norm, grad_y / norm, grad_z / norm), axis=-1)

    return normals

#Depth Refinement Using Normals
def refine_with_normals(depth_map, normals):
    """
    Refines a depth map using surface normals.
    """
    weights = np.linalg.norm(normals, axis=-1)
    weights = cv2.GaussianBlur(weights, (5, 5), sigmaX=1.0)

    refined_depth = depth_map * (1 + 0.1 * weights)
    refined_depth = gaussian_filter(refined_depth, sigma=1.0)

    return refined_depth

#Spatial Attention Module
class SpatialAttention(torch.nn.Module):
    """
    Spatial attention mechanism to enhance depth map features.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention_map = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(attention_map))
        return x * attention_map

#Spatial Attention
def apply_spatial_attention(depth_map):
    """
    Enhances a depth map using spatial attention.
    """
    depth_map_tensor = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spatial_attention = SpatialAttention()
    refined_depth_tensor = spatial_attention(depth_map_tensor)
    refined_depth_map = refined_depth_tensor.squeeze().detach().cpu().numpy()
    return refined_depth_map

def build_3d_object_top_down(image_path, stl_path):
    """
    Constructs a 3D object from an image using depth maps and refinement techniques.
    """
    object_masks = object_detection_with_yolo(image_path)

    #Depth Map Generation
    depth_map = depth_estimation_with_dpt(image_path)

    #Initial Refinement
    refined_depth_map = refine_depth_map(depth_map)

    #Depth-to-Normal Conversion
    normals = depth_to_normal(refined_depth_map)

    #Refine Depth Map Using Normals
    refined_depth_map = refine_with_normals(refined_depth_map, normals)

    #Spatial Attention
    refined_depth_map = apply_spatial_attention(refined_depth_map)

    #Generate Vertices and Faces
    height_map = refined_depth_map * 20
    vertices, faces = [], []
    height, width = height_map.shape
    for y in range(height):
        for x in range(width):
            z = height_map[y, x]
            vertices.append([x * 0.1, y * 0.1, z])
            if x < width - 1 and y < height - 1:
                top_left = y * width + x
                top_right = top_left + 1
                bottom_left = (y + 1) * width + x
                bottom_right = bottom_left + 1
                faces.append([top_left, bottom_left, bottom_right])
                faces.append([top_left, bottom_right, top_right])

    object_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            object_mesh.vectors[i][j] = vertices[face[j]]
    object_mesh.save(stl_path)
    print(f"Refined 3D object saved at: {stl_path}")

uploaded = files.upload()
if uploaded:
    image_path = list(uploaded.keys())[0]
    stl_path = "refined_3d_object.stl"
    build_3d_object_top_down(image_path, stl_path)
    files.download(stl_path)
