export interface DatasetOption {
  value: string;
  label: string;
}

export interface DatasetFormData {
  datasetName: string;
  batchSize: number;
  shuffle: boolean;
  trainSplit: number;
  testSplit: number;
  
  mnistRoot: string;
  mnistTrain: boolean;
  mnistDownload: boolean;
  mnistTransform: string[];
  mnistTransformInput: string;
  
  cifar10Root: string;
  cifar10Train: boolean;
  cifar10Download: boolean;
  cifar10Transform: string[];
  cifar10TransformInput: string;
  
  celebaRoot: string;
  celebaTargetType: string[];
  celebaDownload: boolean;
  celebaTransform: string[];
  celebaTargetTransform: string[];
  celebaTargetTypeInput: string;
  celebaTransformInput: string;
  celebaTargetTransformInput: string;
  
  vocsegmentationRoot: string;
  vocsegmentationImageSet: string;
  vocsegmentationYear: string;
  vocsegmentationDownload: boolean;
  vocsegmentationTransform: string[];
  vocsegmentationTargetTransform: string[];
  vocsegmentationTransformInput: string;
  vocsegmentationTargetTransformInput: string;
  
  csvRoot: string;
  csvFeatureColumns: string[];
  csvLabelColumns: string[];
  csvIsRegressionTask: boolean;
  csvFeatureColumnsInput: string;
  csvLabelColumnsInput: string;
  
  imageFolderRoot: string;
  imageFolderTransform: string[];
  imageFolderAllowEmpty: boolean;
  imageFolderTransformInput: string;
}

export interface CSVPreviewData {
  data: any[];
  columns: string[];
  showingPreview: boolean;
  previewError: string | null;
  file: File | null;
}

export const DATASET_OPTIONS: DatasetOption[] = [
  { value: 'mnist', label: 'MNIST (Handwritten Digits)' },
  { value: 'cifar10', label: 'CIFAR-10 (Colored Images)' },
  { value: 'celeba', label: 'CelebA (Celebrity Faces)' },
  { value: 'vocsegmentation', label: 'VOC Segmentation' },
  { value: 'custom_csv', label: 'Custom CSV Dataset' },
  { value: 'image_folder', label: 'Image Folder Dataset' }
];

export const TRANSFORM_OPTIONS = ['ToTensor', 'Normalize'];

export const CELEBA_TARGET_OPTIONS = [
  'attr', 'identity', 'bbox', 'landmarks'
];

export const VOC_IMAGE_SET_OPTIONS = [
  'train', 'val', 'trainval', 'test'
];

export const VOC_YEAR_OPTIONS = [
  '2007', '2008', '2009', '2010', '2011', '2012'
];

export const DEFAULT_FORM_DATA: DatasetFormData = {
  datasetName: 'mnist',
  batchSize: 32,
  shuffle: true,
  trainSplit: 0.7,
  testSplit: 0.3,
  
  mnistRoot: './data/mnist',
  mnistTrain: true,
  mnistDownload: true,
  mnistTransform: [],
  mnistTransformInput: '',
  
  cifar10Root: './data/cifar10',
  cifar10Train: true,
  cifar10Download: true,
  cifar10Transform: [],
  cifar10TransformInput: '',
  
  // NEW: CelebA defaults
  celebaRoot: './data/celeba',
  celebaTargetType: [],
  celebaDownload: false,
  celebaTransform: [],
  celebaTargetTransform: [],
  celebaTargetTypeInput: '',
  celebaTransformInput: '',
  celebaTargetTransformInput: '',
  
  vocsegmentationRoot: './data/vocsegmentation',
  vocsegmentationImageSet: '',
  vocsegmentationYear: '',
  vocsegmentationDownload: false,
  vocsegmentationTransform: [],
  vocsegmentationTargetTransform: [],
  vocsegmentationTransformInput: '',
  vocsegmentationTargetTransformInput: '',
  
  csvRoot: './data/custom.csv',
  csvFeatureColumns: [],
  csvLabelColumns: [],
  csvIsRegressionTask: false,
  csvFeatureColumnsInput: '',
  csvLabelColumnsInput: '',
  
  imageFolderRoot: './data/images',
  imageFolderTransform: [],
  imageFolderAllowEmpty: false,
  imageFolderTransformInput: ''
};