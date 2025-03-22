import cv2
import os
import torch
import numpy as np
from PIL import Image

%cd 3DDFA-V3-W-O-Args/
from face_box import face_box
from model.recon import face_model
from util.preprocess import get_data_path
from util.io import visualize


def main():
    # Hardcoded settings
    input_folder = 'examples'  # Where images are uploaded
    output_folder = 'examples/results'  # Where results are saved
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize models with minimal config
    recon_model = face_model(backbone='resnet50', device=device)
    face_detector = face_box(detector='retinaface').detector

    # Get list of images from input folder
    image_paths = get_data_path(input_folder)

    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i}: {image_path}")

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        trans_params, img_tensor = face_detector(img)

        # Run reconstruction
        recon_model.input_img = img_tensor.to(device)
        results = recon_model.forward()

        # Create output directory based on image name
        img_name = image_path.split('/')[-1].replace('.png', '').replace('.jpg', '')
        output_dir = os.path.join(output_folder, img_name)
        os.makedirs(output_dir, exist_ok=True)

        # Visualize and save results
        print(results.keys())
        viz = visualize(
            results,
            ldm68=True, ldm106=True, ldm106_2d=True, ldm134=True,
            seg=True, seg_visible=True, useTex=True, extractTex=True
        )
        viz.visualize_and_output(
            trans_params,
            cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR),
            output_dir,
            img_name
        )

if __name__ == '__main__':
    # Colab-specific: Upload images and setup
    from google.colab import files
    uploaded = files.upload()
    os.makedirs('examples', exist_ok=True)
    for filename in uploaded.keys():
        with open(os.path.join('examples', filename), 'wb') as f:
            f.write(uploaded[filename])

    main()
