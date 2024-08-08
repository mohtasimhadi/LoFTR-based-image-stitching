import os
import glob
import torch
import kornia as K
import kornia.feature as KF
from kornia.contrib import ImageStitcher
import matplotlib.pyplot as plt
from termcolor import colored

def print_log(message, level='text'):
    colors = {'Log': 'cyan', 'Warning': 'yellow', 'Error': 'red', 'text': 'white'}
    if level == 'text':
        colored_level = colored(f'     \t', colors.get(level, 'white'))
    else:    
        colored_level = colored(f'[{level}]\t', colors.get(level, 'white'))
    print(f'{colored_level} {message}')

def load_image_paths(folder_path):
    print_log(f"Loading image paths from directory: {folder_path}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        paths = glob.glob(os.path.join(folder_path, ext))
        image_paths.extend(paths)
    print_log(f"Found {len(image_paths)} images.")
    return sorted(image_paths)

def load_images(fnames):
    return [K.io.load_image(fn, K.io.ImageLoadType.RGB32)[None, ...] for fn in fnames]

def stitch_images(image1, image2, temp_folder, i):
    print_log(f"Stitching images: {image1}, {image2}")
    try:
        imgs = load_images([image1, image2])
        IS = ImageStitcher(KF.LoFTR(pretrained="outdoor"), estimator="ransac")
        with torch.no_grad():
            out = IS(*imgs)
            output_path = f'{temp_folder}/{i:04}.png'
            out_image = K.tensor_to_image(out)
            out_image = out_image / 255.0 if out_image.max() > 1.0 else out_image
            plt.imsave(output_path, out_image)
            print_log(f"Stitched image saved at: {output_path}")
    except Exception as e:
        print_log(f"Failed to stitch images {image1} and {image2}: {str(e)}", level='Error')

def iterative_stitching(image_paths, temp_folder):
    total_iterations = 0
    n_images = len(image_paths)
    while n_images > 1:
        total_iterations += n_images // 2
        n_images = (n_images + 1) // 2
    print_log(f"Total iterations needed: {total_iterations}")
    
    iteration = 0
    while len(image_paths) > 1:
        new_image_paths = []
        for i in range(0, len(image_paths) - 1, 2):
            print_log(f"Iteration {iteration}, stitching {len(image_paths)} images.", level='Log')
            stitch_images(image_paths[i], image_paths[i+1], temp_folder, iteration)
            if os.path.exists(f'{temp_folder}/{iteration:04}.png'):
                new_image_paths.append(f'{temp_folder}/{iteration:04}.png')
            iteration += 1
        if len(image_paths) % 2 == 1:
            new_image_paths.append(image_paths[-1])
        image_paths = new_image_paths
    return image_paths[0]

def stitch_all_images_in_directory(directory, temp_folder, iteration_factor=2):
    print_log(f"Stitching all images in directory: {directory}")
    image_paths = load_image_paths(directory)
    image_paths = [image_paths[i] for i in range(0, len(image_paths), iteration_factor)]
    final_image_path = iterative_stitching(image_paths, temp_folder)
    print_log(f"Final stitched image path: {final_image_path}")
    return final_image_path

def main(input_directory, temp_folder, iteration_factor=2):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        print_log(f"Created temporary folder: {temp_folder}")
    final_image_path = stitch_all_images_in_directory(input_directory, temp_folder, iteration_factor)
    print_log(f'Final stitched image saved at: {final_image_path}')

if __name__ == "__main__":
    input_directory = "out/07/18443010518B880E00"
    temp_folder = "out/07/temp"
    main(input_directory, temp_folder, 3)