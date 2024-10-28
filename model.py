import dash
from dash import html, dcc
import dash_uploader as du
from PIL import Image
import io
import base64
import os
import numpy as np

# Define constants for resizing and patch size
TARGET_SIZE = (384, 384)  # DPT often uses 384x384 for ViT models
PATCH_SIZE = (16, 16)  # Size of each patch

# Initialize Dash app
app = dash.Dash(__name__)
upload_folder = "./uploads"
du.configure_upload(app, upload_folder)  # Directory to temporarily save uploads

# Layout
app.layout = html.Div([
    html.H2("DPT Model - Image Upload & Resize with Positional Encoding"),
    du.Upload(
        id="uploader",
        text="Drag and drop or click to upload an image",
        filetypes=["png", "jpg", "jpeg"],
        max_file_size=5  # Max size in MB
    ),
    html.Div(id="output-image")
])

# Image processing function
def process_image(content):
    # Decode base64 image
    _, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))

    # Resize image
    img_resized = img.resize(TARGET_SIZE)

    # Convert resized image back to base64 for display
    buffer = io.BytesIO()
    img_resized.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    return img_resized, f"data:image/png;base64,{encoded_image}"

# Positional encoding function using sine and cosine
def positional_encoding(row, col, depth=64):
    pos_enc = []
    for i in range(depth):
        angle = (row / TARGET_SIZE[0] + col / TARGET_SIZE[1]) / (10000 ** (2 * (i//2 )/ depth))
        pos_enc.append(np.sin(angle) if i % 2 == 0 else np.cos(angle))
    return pos_enc

# Function to split image into patches and get pixel values with positional encoding
def get_patches_and_values(img):
    # Convert image to numpy array
    pixel_values = np.array(img)
    h, w, _ = pixel_values.shape

    patches = []
    patch_values = []
    flattened_patch_values = []  # Store flattened 1D vectors here
    flattened_patch_values_with_encoding = []  # Store flattened 1D vectors with positional encoding

    # Split into patches
    for i in range(0, h, PATCH_SIZE[0]):
        for j in range(0, w, PATCH_SIZE[1]):
            patch = pixel_values[i:i + PATCH_SIZE[0], j:j + PATCH_SIZE[1]]
            patches.append(patch)
            formatted_pixel_values = [f"{{{r}, {g}, {b}}}" for r, g, b in patch.reshape(-1, 3)]
            patch_values.append(formatted_pixel_values)

            # Flatten the patch and store the raw flattened values
            flattened_patch = patch.reshape(-1, 3).flatten().tolist()
            flattened_patch_values.append(flattened_patch)

            # Add positional encoding and store in the encoded list
            pos_enc = positional_encoding(i // PATCH_SIZE[0], j // PATCH_SIZE[1])  # Encode based on patch row and col
            flattened_patch_with_encoding = [x+p for x, p in zip(flattened_patch , pos_enc) ] # Concatenate positional encodings
            flattened_patch_values_with_encoding.append(flattened_patch_with_encoding)

    return patches, patch_values, flattened_patch_values, flattened_patch_values_with_encoding

# Function to simulate multi-head self-attention
def multi_head_self_attention(flattened_vectors):
    # Simulate output by random transformation (you can replace it with a real transformation)
    return np.random.rand(len(flattened_vectors), 128).tolist()  # Example output size (128)

# Function to simulate feed-forward neural network
def feed_forward_nn(attention_output):
    # Example FFNN with 128 neurons in the hidden layer
    weights1 = np.random.rand(128, 64)  # Input size to hidden size
    weights2 = np.random.rand(64, 128)  # Hidden size to output size

    hidden_layer = np.dot(attention_output, weights1)
    output_layer = np.dot(hidden_layer, weights2)
    return output_layer.tolist()

# Function for layer normalization
def layer_normalization(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    return normalized


# Function to perform transposed convolution (upsampling)
# def transposed_convolution(flattened_vectors):
#     # Simulating transposed convolution by reshaping and upsampling
#     upsampled = np.array(flattened_vectors).reshape(-1, 8, 8, 128)  # Example reshape
#     return upsampled.mean(axis=(1, 2)).tolist()  # Example average pooling for simplicity

# Function to perform bilinear interpolation for upsampling
def bilinear_upsampling(image, scale_factor):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Create a new blank image for the upsampled image
    upsampled_image = Image.new("RGB", (new_width, new_height))

    for i in range(new_height):
        for j in range(new_width):
            # Find the corresponding pixel in the original image
            x = j / scale_factor
            y = i / scale_factor

            x1 = int(x)
            x2 = min(x1 + 1, width - 1)
            y1 = int(y)
            y2 = min(y1 + 1, height - 1)

            # Calculate the weights for interpolation
            a = x - x1
            b = y - y1

            # Get the pixel values for the corners
            pixel_1 = np.array(image.getpixel((x1, y1)))
            pixel_2 = np.array(image.getpixel((x2, y1)))
            pixel_3 = np.array(image.getpixel((x1, y2)))
            pixel_4 = np.array(image.getpixel((x2, y2)))

            # Perform bilinear interpolation
            upsampled_pixel = (1 - a) * (1 - b) * pixel_1 + a * (1 - b) * pixel_2 + (1 - a) * b * pixel_3 + a * b * pixel_4

            # Set the pixel in the upsampled image
            upsampled_image.putpixel((j, i), tuple(upsampled_pixel.astype(int)))

    return upsampled_image

# Callback to display resized image, patches, and pixel values with positional encoding
@app.callback(
    dash.dependencies.Output("output-image", "children"),
    [dash.dependencies.Input("uploader", "isCompleted")],
    [dash.dependencies.State("uploader", "fileNames")],
)
def update_output(is_completed, file_names):
    if is_completed and file_names:
        file_path = os.path.join(upload_folder, file_names[0])

        try:
            # Read the original image for display
            with open(file_path, "rb") as image_file:
                original_encoded = base64.b64encode(image_file.read()).decode()
                original_image_src = f"data:image/png;base64,{original_encoded}"

            # Process the resized image and get the image object
            resized_image, resized_image_src = process_image(original_image_src)

            # Get patches and pixel values with positional encoding
            patches, patch_pixel_values, flattened_patch_values, flattened_patch_values_with_encoding = get_patches_and_values(resized_image)

            # Create boxes for patches, pixel values, and flattened vectors with positional encodings
            patch_elements = []
            pixel_value_elements = []
            flattened_vector_elements = []
            flattened_vector_with_encoding_elements = []

            # Create HTML elements for each patch and its pixel values
            for patch, values, flattened_values, flattened_values_with_encoding in zip(
                patches, patch_pixel_values, flattened_patch_values, flattened_patch_values_with_encoding
            ):
                # Convert patch to base64 for display
                patch_image = Image.fromarray(patch)
                buffer = io.BytesIO()
                patch_image.save(buffer, format="PNG")
                encoded_patch = base64.b64encode(buffer.getvalue()).decode()

                patch_elements.append(html.Img(
                    src=f"data:image/png;base64,{encoded_patch}",
                    style={
                        "width": "16px",
                        "height": "16px",
                        "margin": "2px",
                        "border": "1px solid black",
                        "display": "inline-block"
                    }
                ))

                pixel_value_elements.append(html.Div(
                    ", ".join(values),
                    style={
                        "white-space": "nowrap",
                        "margin": "2px",
                        "border": "1px solid black",
                        "padding": "2px",
                        "display": "block"
                    }
                ))

                # Add the flattened 1D vector for each patch without positional encoding
                flattened_vector_elements.append(html.Div(
                    " ".join(map(str, flattened_values)),
                    style={
                        "white-space": "nowrap",
                        "margin": "2px",
                        "border": "1px solid black",
                        "padding": "2px",
                        "display": "block"
                    }
                ))

                # Add the flattened 1D vector for each patch with positional encoding
                flattened_vector_with_encoding_elements.append(html.Div(
                    " ".join(map(str, flattened_values_with_encoding)),
                    style={
                        "white-space": "nowrap",
                        "margin": "2px",
                        "border": "1px solid black",
                        "padding": "2px",
                        "display": "block"
                    }
                ))

            # Get pixel values for the resized image
            resized_pixel_values = [f"{{{r}, {g}, {b}}}" for r, g, b in np.array(resized_image).reshape(-1, 3)]
            
            # Process through the transformer encoder layers (six times)
            final_output = flattened_patch_values_with_encoding
            for _ in range(6):  # Process through six transformer layers
                attention_output = multi_head_self_attention(final_output)
                final_output = feed_forward_nn(attention_output)
            
            # Normalize the output
            normalized_output = layer_normalization(np.array(final_output))
            
            # Perform transposed convolution (upsampling)
            # upsampled_output = transposed_convolution(final_output)
            scale_factor = 2  # Example: upscale by a factor of 2
            upsampled_image = bilinear_upsampling(resized_image, scale_factor)
            
            # Create outputs for attention and feed-forward layers
            attention_output_elements = [
                html.Div(
                    " ".join(map(str, output)),
                    style={
                        "white-space": "nowrap",
                        "margin": "2px",
                        "border": "1px solid black",
                        "padding": "2px",
                        "display": "block"
                    }
                ) for output in attention_output
            ]

            ff_output_elements = [
                html.Div(
                    " ".join(map(str, output)),
                    style={
                        "white-space": "nowrap",
                        "margin": "2px",
                        "border": "1px solid black",
                        "padding": "2px",
                        "display": "block"
                    }
                ) for output in normalized_output.tolist()
            ]
            
            # Create a new div to display the upsampled image
            # upsampled_image = Image.fromarray(np.array(resized_image))  # Just an example; update with actual upsampled output
            buffer = io.BytesIO()
            upsampled_image.save(buffer, format="PNG")
            upsampled_encoded = base64.b64encode(buffer.getvalue()).decode()

            # Display the original image, the resized image, and their pixel values
            return html.Div([
                html.Div([
                    html.H5("Original Image:"),
                    html.Img(src=original_image_src, style={"max-width": "100%", "max-height": "384px", "margin": "10px"})
                ], style={"display": "inline-block", "vertical-align": "top", "margin-right": "20px"}),

                html.Div([
                    html.H5("Resized Image (384x384):"),
                    html.Img(src=resized_image_src, style={"width": "384px", "height": "384px", "border": "1px solid black", "margin": "10px"}),

                    # Box for resized image pixel values
                    html.Div([
                        html.H6("Resized Image Pixel Values:"),
                        html.Div(
                            ", ".join(resized_pixel_values),
                            style={
                                "height": "300px",  # Fixed height
                                "overflow-y": "scroll",  # Enable vertical scrolling
                                "border": "1px solid black",
                                "padding": "10px",
                                "white-space": "pre-wrap",  # Preserve formatting
                                "word-wrap": "break-word"
                            }
                        )
                    ]),

                    # Box for patches
                    html.Div([
                        html.H6("Patches:"),
                        html.Div(
                            html.Div(patch_elements, style={"display": "grid", "gridTemplateColumns": "repeat(24, 16px)", "gap": "2px"}),
                            style={"height": "400px", "overflowY": "scroll", "border": "1px solid black", "padding": "10px", "width": "100%", "margin": "auto"}
                        ),
                    ]),

                    # Box for patch pixel values
                    html.Div([
                        html.H6("Patch Pixel Values:"),
                        html.Div(
                            html.Div(pixel_value_elements, style={"display": "grid", "gridTemplateColumns": "repeat(24, 1fr)", "gap": "2px"}),
                            style={"height": "200px", "overflowY": "scroll", "border": "1px solid black", "padding": "10px", "width": "1600px", "overflowX": "scroll"}
                        ),
                    ]),

                    # Box for flattened 1D vector of each patch without positional encoding
                    html.Div([
                        html.H6("Flattened Patch 1D Vectors without Positional Encoding:"),
                        html.Div(
                            html.Div(flattened_vector_elements, style={"display": "grid", "gridTemplateColumns": "repeat(24, 1fr)", "gap": "2px"}),
                            style={"height": "200px", "overflowY": "scroll", "border": "1px solid black", "padding": "10px", "width": "1600px", "overflowX": "scroll"}
                        ),
                    ]),

                    # Box for flattened 1D vector of each patch with positional encoding
                    html.Div([
                        html.H6("Flattened Patch 1D Vectors with Positional Encoding:"),
                        html.Div(
                            html.Div(flattened_vector_with_encoding_elements, style={"display": "grid", "gridTemplateColumns": "repeat(24, 1fr)", "gap": "2px"}),
                            style={"height": "200px", "overflowY": "scroll", "border": "1px solid black", "padding": "10px", "width": "1600px", "overflowX": "scroll"}
                        ),
                    ]),
                    
                    html.Div([
                        html.H6("attention_output_elements :"),
                        html.Div(
                            html.Div(attention_output_elements, style={"display": "grid", "gridTemplateColumns": "repeat(24, 1fr)", "gap": "2px"}),
                            style={"height": "200px", "overflowY": "scroll", "border": "1px solid black", "padding": "10px", "width": "1600px", "overflowX": "scroll"}
                        ),
                    ]),
                    
                    html.Div([
                        html.H6("ff_output_elements :"),
                        html.Div(
                            html.Div(ff_output_elements, style={"display": "grid", "gridTemplateColumns": "repeat(24, 1fr)", "gap": "2px"}),
                            style={"height": "200px", "overflowY": "scroll", "border": "1px solid black", "padding": "10px", "width": "1600px", "overflowX": "scroll"}
                        ),
                    ]),
                    
                    html.Div([
                    html.H5("Upsampled Image:"),
                    html.Img(src=f"data:image/png;base64,{upsampled_encoded}", style={"max-width": "100%", "max-height": "384px", "margin": "10px"})
                ], style={"display": "inline-block", "vertical-align": "top", "margin-right": "20px"}),

                ], style={"display": "inline-block", "vertical-align": "top"})
            ])
        finally:
            os.remove(file_path)  # Remove uploaded image file after processing

    return html.Div()

if __name__ == "__main__":
    app.run_server(debug=True)
