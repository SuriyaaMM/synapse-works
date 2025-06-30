<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { SET_DATASET_CONFIG } from '$lib/mutations';
  import { GET_MODEL } from '$lib/queries';

  import type { Model, DatasetConfig, MNISTDatasetConfig, 
                DatasetConfigInput, MNISTDatasetConfigInput, 
                CIFAR10DatasetConfigInput, CIFAR10DatasetConfig,
                CustomCSVDatasetConfig, CustomCSVDatasetConfigInput,
                ImageFolderDatasetConfig, ImageFolderDatasetConfigInput } from '../../../../../../source/types';
  import Papa from 'papaparse';
  import './dataset-config.css';

  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let result: Model | null = null;
  let modelDetails: Model | null = null;

  let csvPreviewData: any[] = [];
  let csvColumns: string[] = [];
  let showingPreview = false;
  let previewError: string | null = null;
  let csvFile: File | null = null;

  // Dataset configuration form fields
  let datasetName = 'mnist';
  let batchSize = 32;
  let shuffle = true;
  let trainSplit = 0.7;
  let testSplit = 0.3;

  // MNIST specific variables
  let mnistRoot = './data/mnist';
  let mnistTrain = true;
  let mnistDownload = true;
  let mnistTransform: string[] = [];

  // CIFAR-10 specific variables
  let cifar10Root = './data/cifar10';
  let cifar10Train = true;
  let cifar10Download = true;
  let cifar10Transform: string[] = [];

  // Custom CSV specific variables
  let csvRoot = './data/custom.csv';
  let csvFeatureColumns: string[] = [];
  let csvLabelColumns: string[] = [];
  let csvIsRegressionTask = false;
  let csvFeatureColumnsInput = '';
  let csvLabelColumnsInput = '';

  // Image Folder specific variables
  let imageFolderRoot = './data/images';
  let imageFolderTransform: string[] = [];
  let imageFolderAllowEmpty = false;

  // Transform options
  let mnistTransformInput = '';
  let cifar10TransformInput = '';
  let imageFolderTransformInput = '';

  // Available dataset options
  const datasetOptions = [
    { value: 'mnist', label: 'MNIST (Handwritten Digits)' },
    { value: 'cifar10', label: 'CIFAR-10 (Colored Images)' },
    { value: 'custom_csv', label: 'Custom CSV Dataset' },
    { value: 'image_folder', label: 'Image Folder Dataset' }
  ];

  // Common transform options
  const transformOptions = [
    'ToTensor',
    'Normalize',
    'Resize',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'ColorJitter',
    'RandomCrop',
    'CenterCrop',
    'Grayscale'
  ];

  // Extract modelId from URL path instead of query parameters
  $: {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    if (modelIndex !== -1 && modelIndex + 1 < pathParts.length) {
      modelId = pathParts[modelIndex + 1];
    } else {
      modelId = null;
    }
  }

  // Reactive statement to ensure splits add up to 1.0
  $: {
    if (trainSplit + testSplit !== 1.0) {
      testSplit = parseFloat((1.0 - trainSplit).toFixed(2));
    }
  }

  // Parse comma-separated transform inputs
  $: mnistTransform = mnistTransformInput ? mnistTransformInput.split(',').map(t => t.trim()).filter(t => t) : [];
  $: cifar10Transform = cifar10TransformInput ? cifar10TransformInput.split(',').map(t => t.trim()).filter(t => t) : [];
  $: imageFolderTransform = imageFolderTransformInput ? imageFolderTransformInput.split(',').map(t => t.trim()).filter(t => t) : [];
  $: csvFeatureColumns = csvFeatureColumnsInput ? csvFeatureColumnsInput.split(',').map(t => t.trim()).filter(t => t) : [];
  $: csvLabelColumns = csvLabelColumnsInput ? csvLabelColumnsInput.split(',').map(t => t.trim()).filter(t => t) : [];

  // Fetch model details when modelId changes
  $: if (modelId) {
    fetchModelDetails();
  }

  async function fetchModelDetails() {
    if (!modelId) return;
    
    try {
      const response = await client.query({
        query: GET_MODEL,
        variables: { id: modelId },
        fetchPolicy: 'network-only'
      });
      
      modelDetails = response.data?.getModel;
      
      // Pre-populate form if dataset config already exists
      if (modelDetails?.dataset_config) {
        const config: DatasetConfig = modelDetails.dataset_config;
        datasetName = config.name || 'mnist';
        batchSize = config.batch_size || 32;
        shuffle = config.shuffle !== undefined ? config.shuffle : true;
        
        if (config.split_length && config.split_length.length >= 2) {
          trainSplit = config.split_length[0];
          testSplit = config.split_length[1];
        }
        
        // Handle MNIST config
        const mnistConfig = config as MNISTDatasetConfig;
        if (mnistConfig && datasetName === 'mnist') {
          mnistRoot = mnistConfig.root || './data/mnist';
          mnistTrain = mnistConfig.train !== undefined ? mnistConfig.train : true;
          mnistDownload = mnistConfig.download !== undefined ? mnistConfig.download : true;
          mnistTransformInput = mnistConfig.transform ? mnistConfig.transform.join(', ') : '';
        }

        // Handle CIFAR-10 config
        const cifar10Config = config as CIFAR10DatasetConfig;
        if (cifar10Config && datasetName === 'cifar10') {
          cifar10Root = cifar10Config.root || './data/cifar10';
          cifar10Train = cifar10Config.train !== undefined ? cifar10Config.train : true;
          cifar10Download = cifar10Config.download !== undefined ? cifar10Config.download : true;
          cifar10TransformInput = cifar10Config.transform ? cifar10Config.transform.join(', ') : '';
        }

        // Handle Custom CSV config
        const csvConfig = config as CustomCSVDatasetConfig;
        if (csvConfig && datasetName === 'custom_csv') {
          csvRoot = csvConfig.root || './data/custom.csv';
          csvFeatureColumnsInput = csvConfig.feature_columns ? csvConfig.feature_columns.join(', ') : '';
          csvLabelColumnsInput = csvConfig.label_columns ? csvConfig.label_columns.join(', ') : '';
          csvIsRegressionTask = csvConfig.is_regression_task !== undefined ? csvConfig.is_regression_task : false;
        }

        // Handle Image Folder config
        const imageFolderConfig = config as ImageFolderDatasetConfig;
        if (imageFolderConfig && datasetName === 'image_folder') {
          imageFolderRoot = imageFolderConfig.root || './data/images';
          imageFolderTransformInput = imageFolderConfig.transform ? imageFolderConfig.transform.join(', ') : '';
          imageFolderAllowEmpty = imageFolderConfig.allow_empty !== undefined ? imageFolderConfig.allow_empty : false;
        }
      }
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = 'Failed to fetch model details';
    }
  }

  function validateForm(): string | null {
    if (batchSize <= 0) return 'Batch size must be a positive number';
    if (trainSplit <= 0 || trainSplit >= 1) return 'Train split must be between 0 and 1';
    if (testSplit <= 0 || testSplit >= 1) return 'Test split must be between 0 and 1';
    if (Math.abs(trainSplit + testSplit - 1.0) > 0.001) return 'Train and test splits must sum to 1.0';
    if (!datasetName.trim()) return 'Dataset name is required';
    
    // Dataset-specific validations
    if (datasetName === 'mnist' && !mnistRoot.trim()) return 'MNIST root path is required';
    if (datasetName === 'cifar10' && !cifar10Root.trim()) return 'CIFAR-10 root path is required';
    if (datasetName === 'custom_csv') {
      if (!csvRoot.trim()) return 'CSV root path is required';
      if (csvFeatureColumns.length === 0) return 'At least one feature column is required';
      if (csvLabelColumns.length === 0) return 'At least one label column is required';
    }
    if (datasetName === 'image_folder' && !imageFolderRoot.trim()) return 'Image folder root path is required';
    
    return null;
  }

  async function setDatasetConfig() {
    if (!modelId) {
      error = 'Model ID is missing from URL parameters';
      return;
    }

    const validationError = validateForm();
    if (validationError) {
      error = validationError;
      return;
    }
    
    loading = true;
    error = null;
    result = null;

    try {
      const datasetConfigInput: DatasetConfigInput = {
        name: datasetName,
        batch_size: batchSize,
        shuffle,
        split_length: [trainSplit, testSplit]
      };

      if (datasetName === 'mnist') {
        const mnistConfig: MNISTDatasetConfigInput = {
          root: mnistRoot,
          train: mnistTrain,
          download: mnistDownload,
          transform: mnistTransform.length > 0 ? mnistTransform : undefined
        };
        datasetConfigInput.mnist = mnistConfig;
      }
      else if (datasetName === 'cifar10') {
        const cifar10Config: CIFAR10DatasetConfigInput = {
          root: cifar10Root,
          train: cifar10Train,
          download: cifar10Download,
          transform: cifar10Transform.length > 0 ? cifar10Transform : undefined
        };
        datasetConfigInput.cifar10 = cifar10Config;
      }
      else if (datasetName === 'custom_csv') {
        const csvConfig: CustomCSVDatasetConfigInput = {
          root: csvRoot,
          feature_columns: csvFeatureColumns,
          label_columns: csvLabelColumns,
          is_regression_task: csvIsRegressionTask
        };
        datasetConfigInput.custom_csv = csvConfig;
      }
      else if (datasetName === 'image_folder') {
        const imageFolderConfig: ImageFolderDatasetConfigInput = {
          root: imageFolderRoot,
          transform: imageFolderTransform.length > 0 ? imageFolderTransform : undefined,
          allow_empty: imageFolderAllowEmpty
        };
        datasetConfigInput.image_folder = imageFolderConfig;
      }

      const res = await client.mutate({
        mutation: SET_DATASET_CONFIG,
        variables: { 
          modelId,
          datasetConfig: datasetConfigInput
        }
      });
      
      console.log('Mutation response:', res);
      
      if (!res.data?.setDataset) {
        throw new Error('Failed to set dataset configuration - no data returned');
      }
      
      result = res.data.setDataset;
      
      // Refresh model details
      await fetchModelDetails();
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  function addTransformSuggestion(transformType: 'mnist' | 'cifar10' | 'image_folder', transform: string) {
    if (transformType === 'mnist') {
      const currentTransforms = mnistTransform;
      if (!currentTransforms.includes(transform)) {
        mnistTransformInput = currentTransforms.length > 0 ? `${mnistTransformInput}, ${transform}` : transform;
      }
    } else if (transformType === 'cifar10') {
      const currentTransforms = cifar10Transform;
      if (!currentTransforms.includes(transform)) {
        cifar10TransformInput = currentTransforms.length > 0 ? `${cifar10TransformInput}, ${transform}` : transform;
      }
    } else if (transformType === 'image_folder') {
      const currentTransforms = imageFolderTransform;
      if (!currentTransforms.includes(transform)) {
        imageFolderTransformInput = currentTransforms.length > 0 ? `${imageFolderTransformInput}, ${transform}` : transform;
      }
    }
  }

  async function loadCSVPreview() {
    if (!csvFile) return;
    
    showingPreview = true;
    previewError = null;
    
    try {
      const text = await csvFile.text();
      const results = Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        quoteChar: '"',
        skipFirstNLines: 0,
        preview: 10,
        escapeChar: '\\',
        delimitersToGuess: [',', '\t', '|', ';', Papa.RECORD_SEP, Papa.UNIT_SEP],
        fastMode: undefined,
        transform: (value, header) => {
          if (typeof value === 'string') {
            return value.trim().replace(/^["']+|["']+$/g, '').replace(/[""]/g, '"');
          }
          return value;
        },
      });
      
      if (results.errors.length > 0) {
        throw new Error(`CSV parsing error: ${results.errors[0].message}`);
      }
      
      csvPreviewData = results.data.slice(0, 10);
      csvColumns = results.meta.fields || Object.keys(results.data[0] || {});
      
    } catch (err: any) {
      previewError = err.message;
      csvPreviewData = [];
      csvColumns = [];
    } finally {
      showingPreview = false;
    }
  }

  let selectionMode = 'feature'; // 'feature' or 'label'

  function toggleFeatureColumn(column: string) {
  if (csvFeatureColumns.includes(column)) {
    csvFeatureColumns = csvFeatureColumns.filter(c => c !== column);
  } else {
    csvFeatureColumns = [...csvFeatureColumns, column];
  }
  csvFeatureColumnsInput = csvFeatureColumns.join(', ');
}

function toggleLabelColumn(column : string) {
  if (csvLabelColumns.includes(column)) {
    csvLabelColumns = csvLabelColumns.filter(c => c !== column);
  } else {
    csvLabelColumns = [...csvLabelColumns, column];
  }
  csvLabelColumnsInput = csvLabelColumns.join(', ');
}
</script>

<div class="dataset-config-container">
  <h1 class="dataset-config-heading">Dataset Configuration</h1>

  {#if !modelId}
    <div class="dataset-config-error">
      <p>No model ID provided in the URL.</p>
      <p class="mt-2">
        <a href="/create-model">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <div class="space-y-6">      
      {#if modelDetails}
        <div class="dataset-config-overview">
          <h3 class="font-semibold mb-2">Model Overview</h3>
          <p class="dataset-config-info">
            Model Name: <span>{modelDetails.name}</span>
          </p>
          <p class="dataset-config-info">
            Total Layers: <span>{modelDetails.layers_config?.length || 0}</span>
          </p>
          {#if modelDetails.train_config}
            <p class="dataset-config-info">
              Training: <span>{modelDetails.train_config.epochs} epochs, {modelDetails.train_config.optimizer} optimizer</span>
            </p>
          {:else}
            <p class="dataset-config-warning mt-2">
              ⚠️ No training configuration found. You may want to configure training first.
            </p>
          {/if}
        </div>
      {/if}

      <form on:submit|preventDefault={setDatasetConfig} class="dataset-config-form space-y-6">
        <!-- Dataset Selection -->
        <div>
          <label for="datasetName" class="dataset-config-label">
            Dataset <span class="text-red-500">*</span>
          </label>
          <select
            id="datasetName"
            bind:value={datasetName}
            required
            class="dataset-config-select"
            disabled={loading}
          >
            {#each datasetOptions as option}
              <option value={option.value}>
                {option.label}
              </option>
            {/each}
          </select>
          <p class="dataset-config-description">Choose the dataset for training your model</p>
        </div>

        <!-- General Dataset Settings -->
        <div class="dataset-config-grid">
          <div>
            <label for="batchSize" class="dataset-config-label">
              Batch Size <span class="text-red-500">*</span>
            </label>
            <input
              id="batchSize"
              type="number"
              bind:value={batchSize}
              required
              min="1"
              max="1024"
              class="dataset-config-input"
              disabled={loading}
            />
            <p class="dataset-config-description">Samples per training batch</p>
          </div>

          <div>
            <label for="trainSplit" class="dataset-config-label">
              Train Split <span class="text-red-500">*</span>
            </label>
            <input
              id="trainSplit"
              type="number"
              bind:value={trainSplit}
              step="0.1"
              required
              min="0.1"
              max="0.9"
              class="dataset-config-input"
              disabled={loading}
            />
            <p class="dataset-config-description">Fraction for training</p>
          </div>

          <div>
            <label for="testSplit" class="dataset-config-label">
              Test Split <span class="text-red-500">*</span>
            </label>
            <input
              id="testSplit"
              type="number"
              bind:value={testSplit}
              step="0.1"
              required
              min="0.1"
              max="0.9"
              class="dataset-config-input"
              disabled={loading}
              readonly
            />
            <p class="dataset-config-description">Auto-calculated (1 - train split)</p>
          </div>
        </div>

        <div class="dataset-config-checkbox">
          <input
            id="shuffle"
            type="checkbox"
            bind:checked={shuffle}
            class="h-4 w-4"
            disabled={loading}
          />
          <label for="shuffle" class="ml-2 block text-sm">
            Shuffle dataset
          </label>
          <p class="dataset-config-checkbox-description">Randomize data order during training</p>
        </div>

        <!-- MNIST Specific Settings -->
        {#if datasetName === 'mnist'}
          <div class="dataset-config-section">
            <h3 class="dataset-config-section-title">MNIST Dataset Settings</h3>
            
            <div class="space-y-4">
              <div>
                <label for="mnistRoot" class="dataset-config-label">
                  Data Root Path <span class="text-red-500">*</span>
                </label>
                <input
                  id="mnistRoot"
                  type="text"
                  bind:value={mnistRoot}
                  required
                  placeholder="./data/mnist"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Local path where MNIST data will be stored</p>
              </div>

              <div>
                <label for="mnistTransform" class="dataset-config-label">
                  Transforms (optional)
                </label>
                <input
                  id="mnistTransform"
                  type="text"
                  bind:value={mnistTransformInput}
                  placeholder="ToTensor, Normalize"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Comma-separated list of transforms</p>
                <div class="mt-2 flex flex-wrap gap-2">
                  {#each transformOptions as transform}
                    <button
                      type="button"
                      class="px-2 py-1 text-xs rounded transition-colors {mnistTransform.includes(transform) ? 'bg-blue-200 text-blue-800' : 'bg-gray-200 hover:bg-gray-300'}"
                      on:click={() => addTransformSuggestion('mnist', transform)}
                      disabled={loading || mnistTransform.includes(transform)}
                    >
                      {transform}
                    </button>
                  {/each}
                </div>
              </div>

              <div class="dataset-config-grid">
                <div class="dataset-config-checkbox">
                  <input
                    id="mnistTrain"
                    type="checkbox"
                    bind:checked={mnistTrain}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="mnistTrain" class="ml-2 block text-sm">
                    Use training set
                  </label>
                </div>

                <div class="dataset-config-checkbox">
                  <input
                    id="mnistDownload"
                    type="checkbox"
                    bind:checked={mnistDownload}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="mnistDownload" class="ml-2 block text-sm">
                    Auto-download if missing
                  </label>
                </div>
              </div>
            </div>
          </div>
        {/if}

        <!-- CIFAR-10 Specific Settings -->
        {#if datasetName === 'cifar10'}
          <div class="dataset-config-section">
            <h3 class="dataset-config-section-title">CIFAR-10 Dataset Settings</h3>
            
            <div class="space-y-4">
              <div>
                <label for="cifar10Root" class="dataset-config-label">
                  Data Root Path <span class="text-red-500">*</span>
                </label>
                <input
                  id="cifar10Root"
                  type="text"
                  bind:value={cifar10Root}
                  required
                  placeholder="./data/cifar10"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Local path where CIFAR-10 data will be stored</p>
              </div>

              <div>
                <label for="cifar10Transform" class="dataset-config-label">
                  Transforms (optional)
                </label>
                <input
                  id="cifar10Transform"
                  type="text"
                  bind:value={cifar10TransformInput}
                  placeholder="ToTensor, Normalize, RandomHorizontalFlip"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Comma-separated list of transforms</p>
                <div class="mt-2 flex flex-wrap gap-2">
                  {#each transformOptions as transform}
                    <button
                      type="button"
                      class="px-2 py-1 text-xs rounded transition-colors {cifar10Transform.includes(transform) ? 'bg-blue-200 text-blue-800' : 'bg-gray-200 hover:bg-gray-300'}"
                      on:click={() => addTransformSuggestion('cifar10', transform)}
                      disabled={loading || cifar10Transform.includes(transform)}
                    >
                      {transform}
                    </button>
                  {/each}
                </div>
              </div>

              <div class="dataset-config-grid">
                <div class="dataset-config-checkbox">
                  <input
                    id="cifar10Train"
                    type="checkbox"
                    bind:checked={cifar10Train}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="cifar10Train" class="ml-2 block text-sm">
                    Use training set
                  </label>
                </div>

                <div class="dataset-config-checkbox">
                  <input
                    id="cifar10Download"
                    type="checkbox"
                    bind:checked={cifar10Download}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="cifar10Download" class="ml-2 block text-sm">
                    Auto-download if missing
                  </label>
                </div>
              </div>
            </div>
          </div>
        {/if}

        {#if datasetName === 'custom_csv'}
        <div class="dataset-config-section">
          <h3 class="dataset-config-section-title">Custom CSV Dataset Settings</h3>
          
          <div class="space-y-4">
            <div>
              <label for="csvRoot" class="dataset-config-label">
                CSV File Path <span class="text-red-500">*</span>
              </label>
              <div class="flex space-x-2">
                <input
                  type="file"
                  accept=".csv"
                  on:change={(e) => {
                    const target = e.target as HTMLInputElement | null;
                    csvFile = target && target.files ? target.files[0] : null;
                    if (csvFile) {
                      csvRoot = csvFile.name;
                    }
                  }}
                  class="dataset-config-input"
                  disabled={loading}
                />
                <button
                  type="button"
                  on:click={loadCSVPreview}
                  disabled={loading || showingPreview || !csvRoot.trim()}
                  class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                >
                  {showingPreview ? 'Loading...' : 'Load Preview'}
                </button>
              </div>
              <p class="dataset-config-description">Path to your CSV file</p>
            </div>

            {#if previewError}
              <div class="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                Error loading CSV: {previewError}
              </div>
            {/if}

            {#if csvPreviewData.length > 0}
              <div class="border rounded-lg overflow-hidden">
                <div class="bg-gray-50 px-4 py-2 border-b">
                  <h4 class="font-medium">CSV Preview (First 10 rows)</h4>
                </div>
                <div class="overflow-x-auto max-h-64">
                  <table class="min-w-full text-sm">
                    <thead class="bg-gray-100">
                      <tr>
                        {#each csvColumns as column}
                          <th class="px-3 py-2 text-left font-medium cursor-pointer hover:bg-gray-200 transition-colors select-none
                                    {csvFeatureColumns.includes(column) ? 'bg-green-200 text-green-800' : ''}
                                    {csvLabelColumns.includes(column) ? 'bg-orange-200 text-orange-800' : ''}"
                              on:click={() => toggleFeatureColumn(column)}
                              on:dblclick={() => toggleLabelColumn(column)}
                              tabindex="0"
                              title="Single-click to select as feature, double-click to select as label">
                            {column}
                            {#if csvFeatureColumns.includes(column)}
                              <span class="ml-1 text-xs">(F)</span>
                            {/if}
                            {#if csvLabelColumns.includes(column)}
                              <span class="ml-1 text-xs">(L)</span>
                            {/if}
                          </th>
                        {/each}
                      </tr>
                    </thead>
                    <tbody>
                      {#each csvPreviewData as row, i}
                        <tr class="border-t {i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}">
                          {#each csvColumns as column}
                            <td class="px-3 py-2">{row[column] || ''}</td>
                          {/each}
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              </div>
              <div class="dataset-config-grid">
              <div>
                <label for="csvFeatureColumns" class="dataset-config-label">
                  Feature Columns <span class="text-red-500">*</span>
                </label>
                <input
                  id="csvFeatureColumns"
                  type="text"
                  bind:value={csvFeatureColumnsInput}
                  required
                  placeholder="Single-click column headers above to select features"
                  class="dataset-config-input"
                  disabled={loading}
                  readonly
                />
                <p class="dataset-config-description">Single-click column headers above to select features</p>
              </div>

              <div>
                <label for="csvLabelColumns" class="dataset-config-label">
                  Label Columns <span class="text-red-500">*</span>
                </label>
                <input
                  id="csvLabelColumns"
                  type="text"
                  bind:value={csvLabelColumnsInput}
                  required
                  placeholder="Double-click column headers above to select labels"
                  class="dataset-config-input"
                  disabled={loading}
                  readonly
                />
                <p class="dataset-config-description">Double-click column headers above to select labels</p>
              </div>
            </div>
            {:else if csvColumns.length === 0}
              <p class="text-gray-500 italic">Load CSV preview to see available columns</p>
            {/if}

            <div class="dataset-config-checkbox">
              <input
                id="csvIsRegressionTask"
                type="checkbox"
                bind:checked={csvIsRegressionTask}
                class="h-4 w-4"
                disabled={loading}
              />
              <label for="csvIsRegressionTask" class="ml-2 block text-sm">
                Regression task
              </label>
              <p class="dataset-config-checkbox-description">Check if this is a regression task (unchecked = classification)</p>
            </div>
          </div>
        </div>
      {/if}

        <!-- Image Folder Specific Settings -->
        {#if datasetName === 'image_folder'}
          <div class="dataset-config-section">
            <h3 class="dataset-config-section-title">Image Folder Dataset Settings</h3>
            
            <div class="space-y-4">
              <div>
                <label for="imageFolderRoot" class="dataset-config-label">
                  Images Root Path <span class="text-red-500">*</span>
                </label>
                <input
                  id="imageFolderRoot"
                  type="text"
                  bind:value={imageFolderRoot}
                  required
                  placeholder="./data/images"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Path to folder containing class subdirectories with images</p>
              </div>

              <div>
                <label for="imageFolderTransform" class="dataset-config-label">
                  Transforms (optional)
                </label>
                <input
                  id="imageFolderTransform"
                  type="text"
                  bind:value={imageFolderTransformInput}
                  placeholder="Resize, ToTensor, Normalize"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Comma-separated list of transforms</p>
                <div class="mt-2 flex flex-wrap gap-2">
                  {#each transformOptions as transform}
                    <button
                      type="button"
                      class="px-2 py-1 text-xs rounded transition-colors {imageFolderTransform.includes(transform) ? 'bg-blue-200 text-blue-800' : 'bg-gray-200 hover:bg-gray-300'}"
                      on:click={() => addTransformSuggestion('image_folder', transform)}
                      disabled={loading || imageFolderTransform.includes(transform)}
                    >
                      {transform}
                    </button>
                  {/each}
                </div>
              </div>

              <div class="dataset-config-checkbox">
                <input
                  id="imageFolderAllowEmpty"
                  type="checkbox"
                  bind:checked={imageFolderAllowEmpty}
                  class="h-4 w-4"
                  disabled={loading}
                />
                <label for="imageFolderAllowEmpty" class="ml-2 block text-sm">
                  Allow empty folders
                </label>
                <p class="dataset-config-checkbox-description">Allow class folders with no images</p>
              </div>
            </div>
          </div>
        {/if}

        <div class="flex space-x-3 pt-6">
          <button 
            type="submit"
            disabled={loading}
            class="dataset-config-button"
          >
            {loading ? 'Saving Dataset Config...' : 'Save Dataset Configuration'}
          </button>
        </div>
      </form>

      {#if error}
        <div class="dataset-config-form-error">
          {error}
        </div>
      {/if}

      {#if result}
        <div class="dataset-config-success">
          Dataset Configuration Saved Successfully
        </div>
      {/if}
    </div>
  {/if}
</div>