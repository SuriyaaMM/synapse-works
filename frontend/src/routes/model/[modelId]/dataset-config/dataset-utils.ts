import Papa from 'papaparse';
import type { DatasetFormData, CSVPreviewData } from './dataset-config-types';

// Validation functions
export function validateDatasetForm(formData: DatasetFormData): string | null {
  const { batchSize, trainSplit, testSplit, datasetName } = formData;

  if (batchSize <= 0) return 'Batch size must be a positive number';
  if (trainSplit <= 0 || trainSplit >= 1) return 'Train split must be between 0 and 1';
  if (testSplit <= 0 || testSplit >= 1) return 'Test split must be between 0 and 1';
  if (Math.abs(trainSplit + testSplit - 1.0) > 0.001) return 'Train and test splits must sum to 1.0';
  if (!datasetName.trim()) return 'Dataset name is required';
  
  switch (datasetName) {
    case 'mnist': 
      return !formData.mnistRoot.trim() ? 'MNIST root path is required' : null;
    case 'cifar10': 
      return !formData.cifar10Root.trim() ? 'CIFAR-10 root path is required' : null;
    case 'celeba':
      return !formData.celebaRoot.trim() ? 'CelebA root path is required' : null;
    case 'vocsegmentation':
      return !formData.vocsegmentationRoot.trim() ? 'VOC Segmentation root path is required' : null;
    case 'custom_csv':
      if (!formData.csvRoot.trim()) return 'CSV root path is required';
      if (formData.csvFeatureColumns.length === 0) return 'At least one feature column is required';
      if (formData.csvLabelColumns.length === 0) return 'At least one label column is required';
      return null;
    case 'image_folder': 
      return !formData.imageFolderRoot.trim() ? 'Image folder root path is required' : null;
  }
  return null;
}

export function parseTransformInput(input: string): string[] {
  return input ? input.split(',').map(t => t.trim()).filter(t => t) : [];
}

export function addTransformToInput(currentInput: string, currentTransforms: string[], newTransform: string): string {
  if (currentTransforms.includes(newTransform)) return currentInput;
  return currentTransforms.length > 0 ? `${currentInput}, ${newTransform}` : newTransform;
}

export function toggleArrayItem<T>(array: T[], item: T): T[] {
  return array.includes(item) ? array.filter(i => i !== item) : [...array, item];
}

export function updateTestSplit(trainSplit: number): number {
  return parseFloat((1.0 - trainSplit).toFixed(2));
}

export function createDatasetConfigInput(formData: DatasetFormData) {
  const { datasetName, batchSize, shuffle, trainSplit, testSplit } = formData;
  const config: any = {
    name: datasetName,
    batch_size: batchSize,
    shuffle,
    split_length: [trainSplit, testSplit]
  };

  switch (datasetName) {
    case 'mnist':
      config.mnist = {
        root: formData.mnistRoot,
        train: formData.mnistTrain,
        download: formData.mnistDownload,
        transform: formData.mnistTransform.length > 0 ? formData.mnistTransform : undefined
      };
      break;
    case 'cifar10':
      config.cifar10 = {
        root: formData.cifar10Root,
        train: formData.cifar10Train,
        download: formData.cifar10Download,
        transform: formData.cifar10Transform.length > 0 ? formData.cifar10Transform : undefined
      };
      break;
    case 'celeba':
      config.celeba = {
        root: formData.celebaRoot,
        target_type: formData.celebaTargetType.length > 0 ? formData.celebaTargetType : undefined,
        download: formData.celebaDownload || undefined,
        transform: formData.celebaTransform.length > 0 ? formData.celebaTransform : undefined,
        target_transform: formData.celebaTargetTransform.length > 0 ? formData.celebaTargetTransform : undefined
      };
      break;
    case 'vocsegmentation':
      config.vocsegmentation = {
        root: formData.vocsegmentationRoot,
        image_set: formData.vocsegmentationImageSet || undefined,
        year: formData.vocsegmentationYear || undefined,
        download: formData.vocsegmentationDownload || undefined,
        transform: formData.vocsegmentationTransform.length > 0 ? formData.vocsegmentationTransform : undefined,
        target_transform: formData.vocsegmentationTargetTransform.length > 0 ? formData.vocsegmentationTargetTransform : undefined
      };
      break;
    case 'custom_csv':
      config.custom_csv = {
        root: formData.csvRoot,
        feature_columns: formData.csvFeatureColumns,
        label_columns: formData.csvLabelColumns,
        is_regression_task: formData.csvIsRegressionTask
      };
      break;
    case 'image_folder':
      config.image_folder = {
        root: formData.imageFolderRoot,
        transform: formData.imageFolderTransform.length > 0 ? formData.imageFolderTransform : undefined,
        allow_empty: formData.imageFolderAllowEmpty
      };
      break;
  }
  return config;
}

// CSV utilities
export async function loadCSVPreview(file: File): Promise<CSVPreviewData> {
  if (!file) throw new Error('No file provided');

  try {
    const text = await file.text();
    const results = Papa.parse(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      quoteChar: '"',
      preview: 10,
      escapeChar: '\\',
      delimitersToGuess: [',', '\t', '|', ';', Papa.RECORD_SEP, Papa.UNIT_SEP],
      transform: (value) => typeof value === 'string' ? value.trim().replace(/^["']+|["']+$/g, '').replace(/[""]/g, '"') : value,
    });
    
    if (results.errors.length > 0) {
      throw new Error(`CSV parsing error: ${results.errors[0].message}`);
    }
    
    return {
      data: results.data.slice(0, 10),
      columns: results.meta.fields || Object.keys(results.data[0] || {}),
      showingPreview: false,
      previewError: null,
      file
    };
  } catch (err: any) {
    throw new Error(err.message);
  }
}