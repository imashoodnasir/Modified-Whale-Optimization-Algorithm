% Load images and labels
imageDir = 'path_to_images';
imageFiles = dir(fullfile(imageDir, '*.jpg'));
numImages = numel(imageFiles);

% Preprocessing: Resize and Normalize
imageSize = [256, 256, 3];
augmentedDatastore = augmentedImageDatastore(imageSize, imageFiles, 'ColorPreprocessing', 'gray2rgb');

% Apply Data Augmentation
augmenter = imageDataAugmenter('RandRotation', [0 360], ...
                               'RandXReflection', true, ...
                               'RandYReflection', true, ...
                               'RandXShear', [-10, 10], ...
                               'RandYShear', [-10, 10]);

augmentedDatastore = augmentedImageDatastore(imageSize, imageFiles, 'DataAugmentation', augmenter);

% Load Pretrained Models
squeezeNet = squeezenet;
inceptionResNet = inceptionresnetv2;

% Modify SqueezeNet to Extract Features
layer_squeeze = 'pool10';
featureLayer_squeeze = squeezeNet.Layers(end-3).Name;

% Modify InceptionResNet-V2 to Extract Features
layer_inception = 'avg_pool';
featureLayer_inception = inceptionResNet.Layers(end-2).Name;


