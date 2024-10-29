import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import cv2
import os

class TumorSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasound Tumor Segmentation with Simple Model")
        self.root.configure(bg='#1A1A40')
        self.root.geometry("1200x800")

        # Preload the DCFN+PBAC and U-Net models, and create a simple segmentation model
        self.models = {
            "DFCN+PBAC": self.load_predefined_model("dcfn_pbac_model.keras"),
            "U-Net": self.load_predefined_model("unet_model.keras"),
            "Simple Model": self.create_simple_model()
        }
        self.custom_model = None  # Placeholder for any user-uploaded model
        self.setup_gui()

    def setup_gui(self):
        title_label = tk.Label(self.root, text="Ultrasound Tumor Segmentation Comparison",
                               font=("Arial", 20, "bold"), bg='#1A1A40', fg="white")
        title_label.pack(pady=10)

        # Frame for load buttons
        load_btn_frame = tk.Frame(self.root, bg='#1A1A40')
        load_btn_frame.pack(pady=5)

        # Button to upload a custom model
        custom_model_button = tk.Button(load_btn_frame, text="Upload Model for Comparison",
                                        command=self.load_custom_model,
                                        bg='#004080', fg='white', font=("Arial", 12, "bold"))
        custom_model_button.pack(side=tk.LEFT, padx=5)

        # Image display area
        self.image_panel = tk.Label(self.root, bg='#1A1A40')
        self.image_panel.pack(pady=20)

        load_image_button = tk.Button(self.root, text="Load Ultrasound Image", command=self.load_and_segment_image,
                                      bg='#004080', fg='white', font=("Arial", 14, "bold"))
        load_image_button.pack(pady=15)

        # Area to display results for comparison
        self.result_panel = tk.Label(self.root, bg='#1A1A40', fg="white", font=("Arial", 12))
        self.result_panel.pack(pady=10)

        footer_label = tk.Label(self.root, text="© Tumor-from-Ultrasound-Image - Copyright 2024 | github.com/soheilkooklan",
                                font=("Arial", 10), bg='#1A1A40', fg="white")
        footer_label.pack(side=tk.BOTTOM, pady=10)

    def load_predefined_model(self, model_filename):
        """Loads a predefined model file if it exists."""
        if os.path.exists(model_filename):
            try:
                model = load_model(model_filename)
                print(f"Model {model_filename} loaded successfully.")
                return model
            except Exception as e:
                print(f"Failed to load model {model_filename}: {e}")
        else:
            print(f"Model file {model_filename} not found.")
        return None

    def create_simple_model(self):
        """Creates a simple convolutional segmentation model."""
        input_layer = Input(shape=(128, 128, 1))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        output_layer = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_custom_model(self):
        """Allows the user to upload a custom model for comparison."""
        model_path = filedialog.askopenfilename(filetypes=[("Keras Model Files", "*.keras")])
        if model_path:
            try:
                self.custom_model = load_model(model_path)
                messagebox.showinfo("Model Loaded", "Custom model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load custom model:\n{e}")

    def load_and_segment_image(self):
        """Prompts user to load an ultrasound image and performs segmentation."""
        img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if not img_path:
            return

        # Try to load the ground truth mask file by assuming a naming convention
        mask_path = img_path.replace(".jpg", "_mask.png").replace(".jpeg", "_mask.png").replace(".png", "_mask.png")
        if not os.path.exists(mask_path):
            messagebox.showwarning("Missing Mask", "No ground truth mask found for this image.")
            return

        # Load the image and mask
        self.loaded_img = Image.open(img_path)
        self.loaded_mask = Image.open(mask_path).convert("L")
        self.loaded_img = self.preprocess_ultrasound_image(self.loaded_img)

        # Display the loaded image
        img_tk = ImageTk.PhotoImage(self.loaded_img)
        self.image_panel.configure(image=img_tk)
        self.image_panel.image = img_tk

        # Perform segmentation with the simple model and display results
        segmented_img, accuracy = self.segment_with_model(self.models["Simple Model"])
        self.display_segmented_image(segmented_img, f"Simple Model Result (DSC: {accuracy:.2f}%)")

        # Perform segmentation with DCFN+PBAC and U-Net for comparison
        self.compare_segmentation()

    def preprocess_ultrasound_image(self, image):
        grayscale_img = image.convert("L")
        grayscale_img = grayscale_img.resize((256, 256), Image.LANCZOS)
        img_array = np.array(grayscale_img)

        # Apply median and Gaussian filters to reduce speckle noise
        img_array = cv2.medianBlur(img_array, 3)
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        return Image.fromarray(img_array)

    def compare_segmentation(self):
        results = []

        # Run DCFN+PBAC and U-Net models
        for model_name in ["DFCN+PBAC", "U-Net"]:
            if self.models[model_name]:
                segmented_img, accuracy = self.segment_with_model(self.models[model_name])
                self.display_segmented_image(segmented_img, f"{model_name} Result (DSC: {accuracy:.2f}%)")
                results.append(f"{model_name}: DSC = {accuracy:.2f}%")

        # If a custom model is loaded, compare it with the DCFN+PBAC and U-Net
        if self.custom_model:
            segmented_img, accuracy = self.segment_with_model(self.custom_model)
            self.display_segmented_image(segmented_img, f"Custom Model Result (DSC: {accuracy:.2f}%)")
            results.append(f"Custom Model: DSC = {accuracy:.2f}%")

        # Update GUI with comparison results
        self.result_panel.config(text="\n".join(results))

    def dice_coefficient(self, pred, truth):
        """Calculates the Dice Similarity Coefficient (DSC) between prediction and ground truth."""
        intersection = np.sum(pred * truth)
        return (2. * intersection) / (np.sum(pred) + np.sum(truth)) if (np.sum(pred) + np.sum(truth)) > 0 else 1.0

    def segment_with_model(self, model):
        """Segments the tumor in the loaded image using the provided model and calculates accuracy."""
        img_array = np.array(self.loaded_img) / 255.0
        img_array = cv2.resize(img_array, (128, 128))
        img_array = np.expand_dims(img_array, axis=(0, -1))

        # Predict segmentation mask
        pred_mask = model.predict(img_array)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (256, 256))

        # Calculate Dice Similarity Coefficient with ground truth
        ground_truth = np.array(self.loaded_mask.resize((256, 256))) / 255.0
        ground_truth = (ground_truth > 0.5).astype(np.uint8)
        accuracy = self.dice_coefficient(pred_mask, ground_truth) * 100  # Convert to percentage

        return Image.fromarray((pred_mask * 255).astype(np.uint8), 'L'), accuracy

    def display_segmented_image(self, segmented_img, title):
        """Displays each model's segmented image side by side."""
        segmented_img = ImageTk.PhotoImage(segmented_img)
        segmented_window = tk.Toplevel(self.root)
        segmented_window.title(title)
        segmented_window.geometry("300x300")
        img_label = tk.Label(segmented_window, image=segmented_img)
        img_label.image = segmented_img
        img_label.pack()
        segmented_window.mainloop()

# Initialize and run the application
root = tk.Tk()
app = TumorSegmentationApp(root)
root.mainloop()