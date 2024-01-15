# Road Image Segmentation from Satellite Images

This project performs semantic segmentation of road networks from satellite images using deep learning. A custom U-Net-like convolutional neural network is trained using PyTorch to predict road masks from satellite imagery.

## Dataset

The [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) is used for training. The dataset contains pairs of satellite images and their corresponding road masks:

- **Input Images**: RGB satellite images of size `256x256`.
- **Ground Truth Masks**: Binary masks where roads are represented as white pixels (`1`) and non-road areas as black pixels (`0`).

Sample data shape:
- `X_train`: `(512, 3, 256, 256)` — 512 RGB images.
- `Y_train`: `(512, 1, 256, 256)` — Corresponding road masks.

## Model Architecture

The model is a U-Net-inspired convolutional network:

1. **Encoder**: Sequential convolutional blocks with downsampling via max pooling.
2. **Bottleneck**: A bottleneck in the middle.
3. **Decoder**: Transposed convolutions with skip connections to upsample the feature maps.

Some Dropout layers are added as regularizers.

**Model Input and Output**

- Input channels: 3 (RGB)
- Output channels: 1 (Binary segmentation mask)

## Training

### Configuration
- **Optimizer**: Adam with a learning rate of `0.001`.
- **Loss Function**: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`).
- **Batch Size**: 32

### Training Process
The training script iterates through the dataset in batches, computes the loss, and updates model parameters. The training loss is logged after each epoch and visualized as a line plot.

Run the training script:

## Results

After training, the model provides reasonable predictions for road segmentation. See the notebook for provided visualizations.
